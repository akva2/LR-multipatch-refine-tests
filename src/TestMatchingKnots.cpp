#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "LRSpline/LRSplineSurface.h"
#include "GoTools/geometry/SplineSurface.h"
#include "GoTools/geometry/ObjectHeader.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace std;
using namespace LR;
using namespace Go;

// read multipatch g2 file and return LRSplineSurface representation of all patches
vector<LRSplineSurface*> readFile(const string &filename) {
  vector<LRSplineSurface*> results;
  ObjectHeader head;
  SplineSurface ss;

  ifstream in(filename);
  if(!in.good()) {
    cerr << "could not open file: " << filename << endl;
    return results;
  }

  while(!in.eof()) { // for all patches
    in >> head >> ss;
    results.push_back(new LRSplineSurface(&ss));
    ws(in); // eat whitespaces
  }

  for(auto lr : results) {
    lr->generateIDs();
    // disable all a posteriori mesh fixes (makes them unpredictable)
    lr->setCloseGaps(false);
    lr->setMaxTjoints(-1);
    lr->setMaxAspectRatio(1e9, false);
  }
  return results;
}

// write multipatch lr model to file for manual inspection
bool writeFile(const string &filename, vector<LRSplineSurface*> model) {
  ofstream out(filename);
  for(auto lr : model)
    out << *lr << endl;
}

// check that new knots contain all original knots (one-way matching)
static bool check_orig_knots_is_subset(const std::vector<double>& knots, const std::vector<double>& new_knots) {
  for (auto& it : new_knots)
    if (std::find_if(knots.begin(), knots.end(), [it](double a){return fabs(it-a) < 1e-4;}) == knots.end())
      return false;

  return true;
}

// check that knots vectors are identical (two-way matching)
static bool check_matching_knots(const std::vector<double>& knots1, const std::vector<double>& knots2) {
  if(knots1.size() != knots2.size())
    return false;
  for(size_t i=0; i<knots1.size(); ++i)
    if(fabs(knots1[i]-knots2[i]) > 1e-4)
      return false;

  return true;
}

// check that all elements continue to be perfect squares (isotropic / basis-function refinement)
static bool check_isotropic_refinement(LRSplineSurface *lr) {
  for(auto el : lr->getAllElements())
    if(fabs( (el->umax() - el->umin()) - (el->vmax() - el->vmin()) ) >1e-4)
      return false;

  return true;
}

// Refinement strategy 1: refine all 4 corners of the LAST patch
static void refine1(vector<LRSplineSurface*> model) {
  vector<Basisfunction*> oneFunction;
  vector<int> corner_idx;
  // fetch index of all 4 corners
  model.back()->getEdgeFunctions(oneFunction, SOUTH_WEST);
  corner_idx.push_back(oneFunction[0]->getId());
  model.back()->getEdgeFunctions(oneFunction, SOUTH_EAST);
  corner_idx.push_back(oneFunction[0]->getId());
  model.back()->getEdgeFunctions(oneFunction, NORTH_WEST);
  corner_idx.push_back(oneFunction[0]->getId());
  model.back()->getEdgeFunctions(oneFunction, NORTH_EAST);
  corner_idx.push_back(oneFunction[0]->getId());
  // do the actual refinement
  model.back()->refineBasisFunction(corner_idx);
}

// Refinement strategy 2: randomly one function on all patches independently
static void refine2(vector<LRSplineSurface*> model) {
  for(size_t i=0; i<model.size(); ++i)
    model[i]->refineBasisFunction((13*i+17) % model[i]->nBasisFunctions());
}

// Geometry 2: lshape
static vector<LRSplineSurface*> geom2() {
  return readFile("../geometries/lshape.g2");
}

static void fix2(vector<LRSplineSurface*> &lr) {
  // match patch 1 to patch 2
  vector<double> knots3 = lr[1]->getEdgeKnots(SOUTH, true);
  vector<double> knots4 = lr[2]->getEdgeKnots(NORTH, true);
  lr[1]->matchParametricEdge(SOUTH, knots4, true);
  lr[2]->matchParametricEdge(NORTH, knots3, true);

  // match patch 0 to patch 1
  vector<double> knots1 = lr[0]->getEdgeKnots(EAST, true);
  vector<double> knots2 = lr[1]->getEdgeKnots(WEST, true);
  lr[0]->matchParametricEdge(EAST, knots2, true);
  lr[1]->matchParametricEdge(WEST, knots1, true);
}

static int check2(vector<LRSplineSurface*> &lr) {
  // check that it all worked well
  if(! check_matching_knots(lr[0]->getEdgeKnots(EAST, true),
                            lr[1]->getEdgeKnots(WEST, true)))
    return 1;
  if(! check_matching_knots(lr[1]->getEdgeKnots(SOUTH, true),
                            lr[2]->getEdgeKnots(NORTH, true)))
    return 1;
  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
}


// Geometry 3: Star-shaped 6-patch geometry
static vector<LRSplineSurface*> geom3() {
  return readFile("../geometries/star.g2");
}

static void fix3(vector<LRSplineSurface*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    for(int i=0; i<6; ++i) {
      int j = (i+1)%6;
      vector<double> knots1 = lr[i]->getEdgeKnots(WEST,  true);
      vector<double> knots2 = lr[j]->getEdgeKnots(SOUTH, true);
      change |= lr[i]->matchParametricEdge(WEST,  knots2, true);
      change |= lr[j]->matchParametricEdge(SOUTH, knots1, true);
    }
  }
}
static int check3(const vector<LRSplineSurface*> &lr) {
  for(int i=0; i<6; ++i) {
    int j = (i+1)%6;
    if(! check_matching_knots(lr[i]->getEdgeKnots(WEST,  true),
                              lr[j]->getEdgeKnots(SOUTH, true)))
      return 1;
    if(! check_isotropic_refinement(lr[i]))
      return 1;
  }
}

// 2 patch, flip + reverse
static int case1() {
#ifdef HAVE_MPI
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank > 1)
    return 0;

  // create object
  std::unique_ptr<LRSplineSurface> lr;
  if (rank == 0) {
    lr.reset(new LRSplineSurface(5,5,4,4));
    lr->refineElement(0);
    lr->refineElement(4);
    lr->refineElement(6);
    lr->refineElement(4);
    vector<double> knots = lr->getEdgeKnots(WEST, true);
    cout << "Edge nodes on WEST side:\n";
    for(auto d : knots)
      cout << d << endl;
    // manually reverse knots (0,1) -> (1,0)
    for(auto& it : knots)
      it = 1.0-it;
    int size = knots.size();
    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    ofstream out("lr1.eps");
    lr->writePostscriptMesh(out);
    out.close();
  } else {
    lr.reset(new LRSplineSurface(5,5,4,4));
    int size;
    MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots(size);
    MPI_Recv(knots.data(), size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    lr->matchParametricEdge(NORTH, knots, true);
    cout << *lr << std::endl;
    auto new_knots = lr->getEdgeKnots(NORTH, true);
    if (!check_orig_knots_is_subset(knots, new_knots))
      return 1;
    ofstream out("lr2.eps");
    lr->writePostscriptMesh(out);
    out.close();
  }
#else
  // create objects
  LRSplineSurface lr1(5,5,4,4);
  LRSplineSurface lr2(5,5,4,4);

  // do some refinements
  lr1.refineElement(0);
  lr1.refineElement(4);
  lr1.refineElement(6);
  lr1.refineElement(4);
  vector<double> knots = lr1.getEdgeKnots(WEST, true);
  cout << "Edge nodes on WEST side:\n";
  for(auto d : knots)
    cout << d << endl;
  // manually reverse knots (0,1) -> (1,0)
  for(int i=0; i<knots.size(); i++)
    knots[i] = 1-knots[i];

  // match on other side
  lr2.matchParametricEdge(NORTH, knots, true);
  cout << lr2 << endl;

  auto new_knots = lr2.getEdgeKnots(NORTH, true);
  if (!check_orig_knots_is_subset(knots, new_knots))
    return 1;

  // dump files for easy inspection
  ofstream out1("lr1.eps");
  lr1.writePostscriptMesh(out1);
  ofstream out2("lr2.eps");
  lr2.writePostscriptMesh(out2);
  out1.close();
  out2.close();
#endif

  return 0;
}

// 3 patch, Lshape
static int case2() {
#ifdef HAVE_MPI
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank > 2)
    return 0;

  vector<LRSplineSurface*> lr = readFile("../geometries/lshape.g2");
  if (rank == 0) {
    vector<Basisfunction*>   oneFunction;
    lr[0]->getEdgeFunctions( oneFunction, SOUTH_EAST );
    lr[0]->refineBasisFunction( oneFunction[0]->getId() );
    vector<double> knots1 = lr[0]->getEdgeKnots(EAST, true);
    int size = knots1.size();

    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots1.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);

    MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots2(size);
    MPI_Recv(knots2.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    lr[0]->matchParametricEdge(EAST, knots2, true);

    MPI_Recv(&size, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    knots2.resize(size);
    MPI_Recv(knots2.data(), size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    knots1 = lr[0]->getEdgeKnots(EAST, true);
    if (!check_matching_knots(knots1, knots2))
      return 1;
    if (!check_isotropic_refinement(lr[0]))
      return 1;
    ofstream out("lr1.eps");
    lr[0]->writePostscriptMesh(out);
    out.close();
  } else if (rank == 1) {
    int size;
    MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots1(size);
    MPI_Recv(knots1.data(), size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots2 = lr[1]->getEdgeKnots(WEST, true);
    size = knots2.size();

    MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    lr[1]->matchParametricEdge(WEST, knots1, true);
    vector<double> knots3 = lr[1]->getEdgeKnots(SOUTH, true);
    size = knots3.size();

    MPI_Send(&size, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
    MPI_Send(knots3.data(), size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD);

    MPI_Recv(&size, 1, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots4(size);
    MPI_Recv(knots4.data(), size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    lr[1]->matchParametricEdge(SOUTH, knots4, true);

    knots2 = lr[1]->getEdgeKnots(WEST, true);
    size = knots2.size();
    MPI_Send(&size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);

    MPI_Recv(&size, 1, MPI_INT, 2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    knots4.resize(size);
    MPI_Recv(knots4.data(), size, MPI_DOUBLE, 2, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    knots3 = lr[1]->getEdgeKnots(SOUTH, true);
    if (!check_matching_knots(knots3, knots4))
      return 1;
    if (!check_isotropic_refinement(lr[1]))
      return 1;
    ofstream out("lr2.eps");
    lr[1]->writePostscriptMesh(out);
    out.close();
  } else {
    int size;
    MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots3(size);
    MPI_Recv(knots3.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<double> knots4 = lr[2]->getEdgeKnots(NORTH, true);
    size = knots4.size();
    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots4.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    lr[2]->matchParametricEdge(NORTH, knots3, true);

    knots4 = lr[2]->getEdgeKnots(NORTH, true);
    size = knots4.size();
    MPI_Send(&size, 1, MPI_INT, 1, 3, MPI_COMM_WORLD);
    MPI_Send(knots4.data(), size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);

    if (!check_isotropic_refinement(lr[2]))
      return 1;
    ofstream out("lr3.eps");
    lr[2]->writePostscriptMesh(out);
    out.close();
  }
#else
  vector<LRSplineSurface*> lr = readFile("../geometries/lshape.g2");
  vector<Basisfunction*>   oneFunction ;
  lr[0]->getEdgeFunctions( oneFunction, SOUTH_EAST );

  // do a corner refinement at the kink of the L-shape (patch 0)
  lr[0]->refineBasisFunction( oneFunction[0]->getId() );

  // match patch 0 to patch 1
  vector<double> knots1 = lr[0]->getEdgeKnots(EAST, true);
  vector<double> knots2 = lr[1]->getEdgeKnots(WEST, true);
  lr[0]->matchParametricEdge(EAST, knots2, true);
  lr[1]->matchParametricEdge(WEST, knots1, true);

  // match patch 1 to patch 2
  vector<double> knots3 = lr[1]->getEdgeKnots(SOUTH, true);
  vector<double> knots4 = lr[2]->getEdgeKnots(NORTH, true);
  lr[1]->matchParametricEdge(SOUTH, knots4, true);
  lr[2]->matchParametricEdge(NORTH, knots3, true);

  // check that it all worked well
  if(! check_matching_knots(lr[0]->getEdgeKnots(EAST, true),
                            lr[1]->getEdgeKnots(WEST, true)))
    return 1;
  if(! check_matching_knots(lr[1]->getEdgeKnots(SOUTH, true),
                            lr[2]->getEdgeKnots(NORTH, true)))
    return 1;
  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;

  // print results to file for manual debugging
  writeFile("mesh.lr", lr);
#endif

  return 0;
}

// 6 patch, star
static int case3() {
  vector<LRSplineSurface*> lr = readFile("../geometries/star.g2");
#ifdef HAVE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank > 5)
    return 0;

  lr[rank]->refineBasisFunction(3+rank);

  int change = 1;
  while (change) {
    change = 0;
    vector<double> knots1 = lr[rank]->getEdgeKnots(WEST, true);
    vector<double> knots2 = lr[rank]->getEdgeKnots(SOUTH, true);
    if (rank == 0) {
      // recv south from 1
      int size;
      MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector<double> knots3(size);
      MPI_Recv(knots3.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change += lr[0]->matchParametricEdge(WEST,  knots3, true);

      // recv west from 5
      MPI_Recv(&size, 1, MPI_INT, 5, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector<double> knots4(size);
      MPI_Recv(knots4.data(), size, MPI_DOUBLE, 5, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change += lr[0]->matchParametricEdge(SOUTH,  knots4, true);

      // send west to 1
      size = knots1.size();
      MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
      MPI_Send(knots1.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);

      // send west to 5
      size = knots2.size();
      MPI_Send(&size, 1, MPI_INT, 5, 1, MPI_COMM_WORLD);
      MPI_Send(knots2.data(), size, MPI_DOUBLE, 5, 2, MPI_COMM_WORLD);
    } else {
      // send south knots
      int size = knots2.size();
      MPI_Send(&size, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
      MPI_Send(knots2.data(), size, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);

      // send west knots
      size = knots1.size();
      MPI_Send(&size, 1, MPI_INT, (rank+1)%6, 1, MPI_COMM_WORLD);
      MPI_Send(knots1.data(), size, MPI_DOUBLE, (rank+1)%6, 2, MPI_COMM_WORLD);

      // recv south knots
      MPI_Recv(&size, 1, MPI_INT, (rank+1)%6, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      knots2.resize(size);
      MPI_Recv(knots2.data(), size, MPI_DOUBLE, (rank+1)%6, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // recv west knots
      MPI_Recv(&size, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      knots1.resize(size);
      MPI_Recv(knots1.data(), size, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change += lr[rank]->matchParametricEdge(WEST,  knots2, true);
      change += lr[rank]->matchParametricEdge(SOUTH,  knots1, true);
    }
    change += lr[rank]->matchParametricEdge(WEST,  knots2, true);

    int change2 = change;
    MPI_Allreduce(&change2, &change, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  vector<double> knots1 = lr[rank]->getEdgeKnots(WEST, true);
  vector<double> knots2 = lr[rank]->getEdgeKnots(SOUTH, true);
  if (rank == 0) {
    int size = knots2.size();
    MPI_Send(&size, 1, MPI_INT, 5, 1, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, 5, 2, MPI_COMM_WORLD);

    MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    knots2.resize(size);
    MPI_Recv(knots2.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int size;
    MPI_Recv(&size, 1, MPI_INT, (rank+1)%6, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots3(size);
    MPI_Recv(knots3.data(), size, MPI_DOUBLE, (rank+1)%6, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size = knots2.size();
    MPI_Send(&size, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);

    knots2 = knots3;
  }

  if (!check_matching_knots(knots1, knots2))
    return 1;

  if(! check_isotropic_refinement(lr[rank]))
    return 1;

  std::stringstream str;
  str << "lr" << rank << ".lr";
  ofstream out(str.str());
  out << *lr[rank];
  out.close();

#else
  lr[0]->refineBasisFunction(3);
  lr[1]->refineBasisFunction(4);
  lr[2]->refineBasisFunction(5);
  lr[3]->refineBasisFunction(6);
  lr[4]->refineBasisFunction(7);
  lr[5]->refineBasisFunction(8);

  bool change = true;
  while(change) {
    change = false;
    for(int i=0; i<6; ++i) {
      int j = (i+1)%6;
      vector<double> knots1 = lr[i]->getEdgeKnots(WEST,  true);
      vector<double> knots2 = lr[j]->getEdgeKnots(SOUTH, true);
      change |= lr[i]->matchParametricEdge(WEST,  knots2, true);
      change |= lr[j]->matchParametricEdge(SOUTH, knots1, true);
    }
  }

  // check that it all worked well
  for(int i=0; i<6; ++i) {
    int j = (i+1)%6;
    if(! check_matching_knots(lr[i]->getEdgeKnots(WEST,  true),
                              lr[j]->getEdgeKnots(SOUTH, true)))
      return 1;
    if(! check_isotropic_refinement(lr[i]))
      return 1;
  }

  // print results to file for manual debugging
  writeFile("mesh.lr", lr);
#endif

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "File usage: " << argv[0] << "<geometry> <refinement> <iterations>" << std::endl;
    std::cerr << "  geometry: " << std::endl;
    std::cerr << "    2 = L-shape, 3 patches" << std::endl;
    std::cerr << "    3 = Star, 6 patches" << std::endl;
    std::cerr << "    4 = Self-loop, 2 patches" << std::endl;
    std::cerr << "  refinement: " << std::endl;
    std::cerr << "    1 = Corner refinement (all corners on last patch)" << std::endl;
    std::cerr << "    2 = One random function on each patch (chosen independently)" << std::endl;
    std::cerr << "  iterations: (int) number of refinement that is to be performed" << std::endl;
    return 1;
  }

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  vector<LRSplineSurface*> model;
  int geometry   = atoi(argv[1]);
  int refinement = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  if(geometry == 2)
    model = geom2();
  else if(geometry == 3)
    model = geom3();
  else
    return 1;

  for(int i=0; i<iterations; i++) {
    if(refinement == 1)
      refine1(model);
    else if(refinement == 2)
      refine2(model);

    if(geometry == 2)
      fix2(model);
    else if(geometry == 3)
      fix3(model);
    for(auto lr : model) {
      lr->generateIDs();
    }
  }

  int result = 1;
  if(geometry == 2)
    result = check2(model);
  else if(geometry == 3)
    result = check3(model);

  // print results to file for manual debugging
  writeFile("mesh.lr", model);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return result;
}
