#include <iostream>
#include <fstream>
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

  for(auto lr : results)
    lr->generateIDs();
  return results;
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

// 2 patch, flip + reverse
static int case1() {
#ifdef USE_MPI
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank > 1)
    return 0;

  // create object
  std::unique_ptr<LRSplineSurface> lr;
  if (rank == 0) {
    lr1.reset(new LRSplineSurface(5,5,4,4));
    lr.refineElement(0);
    lr.refineElement(4);
    lr.refineElement(6);
    lr.refineElement(4);
    vector<double> knots = lr->getEdgeKnots(WEST, true);
    cout << "Edge nodes on WEST side:\n";
    for(auto d : knots)
      cout << d << endl;
    // manually reverse knots (0,1) -> (1,0)
    for(auto& it : knots)
      it = 1.0-it;
    int size = knots.size();
    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots.data(), size, MP_DOUBLE, 1, 2, MPI_COMM_WORLD);
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

  return 0;
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

  return 0;
#endif
}

// 3 patch, Lshape
static int case2() {
  vector<LRSplineSurface*> lr          = readFile("../geometries/lshape.g2");
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

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Need a parameter - the test to run" << std::endl;
    std::cerr << "1 = 2 patch, flip + reverse" << std::endl;
    return 1;
  }

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  int result = 1;
  if (atoi(argv[1]) == 1)
    result = case1();
  else if (atoi(argv[1]) == 2)
    result = case2();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return result;
} 
