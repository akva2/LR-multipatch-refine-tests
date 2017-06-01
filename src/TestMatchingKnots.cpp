#include <iostream>
#include <sstream>
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

// serial checking of global model
typedef int  (*check_function)( const vector<LRSplineSurface*>&);

// refinement, geometry and topology functions related to each case (SERIAL)
typedef void                     (*refine_function)(const vector<LRSplineSurface*>&);
typedef void                     (*fix_function)(         vector<LRSplineSurface*>&);
typedef vector<LRSplineSurface*> (*geom_function)();

// refinement, geometry and topology functions related to each case (MPI)
typedef void             (*mpiref_function)( int, LRSplineSurface*);
typedef void             (*mpifix_function)( int, LRSplineSurface*);
typedef LRSplineSurface* (*mpigeom_function)(int);

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
void writeFile(const string &filename, vector<LRSplineSurface*> model) {
  if(model.size() == 0) return;
  ofstream out(filename);
  for(auto lr : model)
    out << *lr << endl;
}

/***************** ASSERTION FUNCTIONS  ******************
 *
 * logical tests to see if the resulting multipatch model
 * is indeed matching.
 *
 *********************************************************/

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

/***************** REFINEMENT FUNCTIONS  *****************
 *
 * different refinement schemes that is used to test all
 * geometries.
 *
 *********************************************************/

// Refinement strategy 1: refine all 4 corners of the FIRST patch
static void refine1(const vector<LRSplineSurface*>& model) {
  vector<Basisfunction*> oneFunction;
  vector<int> corner_idx;
  // fetch index of all 4 corners
  model[0]->getEdgeFunctions(oneFunction, SOUTH_WEST);
  corner_idx.push_back(oneFunction[0]->getId());
  model[0]->getEdgeFunctions(oneFunction, SOUTH_EAST);
  corner_idx.push_back(oneFunction[0]->getId());
  model[0]->getEdgeFunctions(oneFunction, NORTH_WEST);
  corner_idx.push_back(oneFunction[0]->getId());
  model[0]->getEdgeFunctions(oneFunction, NORTH_EAST);
  corner_idx.push_back(oneFunction[0]->getId());
  // do the actual refinement
  model[0]->refineBasisFunction(corner_idx);
}

// Refinement strategy 1: MPI implementation
static void mpiref1(int rank, LRSplineSurface *patch) {
  if(rank > 0) return; // skip all but the first patch
  refine1(vector<LRSplineSurface*>(1, patch));
}

// Refinement strategy 2: randomly one function on all patches independently
static void refine2(const vector<LRSplineSurface*>& model) {
  for(size_t i=0; i<model.size(); ++i)
    model[i]->refineBasisFunction((13*i+17) % model[i]->nBasisFunctions());
}

// Refinement strategy 2: MPI implementation
static void mpiref2(int rank, LRSplineSurface *patch) {
  patch->refineBasisFunction((13*rank+17) % patch->nBasisFunctions());
}

/***************** Case 2: L-shape  **********************
 *
 * +--------+--------+
 * |        |        |
 * |    #1  |   #2   |
 * |        |        |
 * +--------+--------+
 *          |        |
 *          |   #3   |   <--- this guy is p=(3,2)
 *          |        |
 *          +--------+
 *
 *********************************************************/

static vector<LRSplineSurface*> geom2() {
  return readFile("../geometries/lshape.g2");
}

static LRSplineSurface* mpigeom2(int rank) {
  vector<LRSplineSurface*> allPatches = geom2();
  return allPatches[rank]->copy();
}

static void fix2(vector<LRSplineSurface*> &lr) {
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
}

static void mpifix2(int rank, LRSplineSurface* lr) {
#ifdef HAVE_MPI
  if(rank == 0) {
    vector<double> knots1 = lr->getEdgeKnots(EAST, true);
    int size = knots1.size();
    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots1.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots2(size);
    MPI_Recv(knots2.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    lr->matchParametricEdge(EAST, knots2, true);

    MPI_Recv(&size, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    knots2.resize(size);
    MPI_Recv(knots2.data(), size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if(rank == 1) {
    int size;
    MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots1(size);
    MPI_Recv(knots1.data(), size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots2 = lr->getEdgeKnots(WEST, true);
    size = knots2.size();
    MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    lr->matchParametricEdge(WEST, knots1, true);
    vector<double> knots3 = lr->getEdgeKnots(SOUTH, true);
    size = knots3.size();
    MPI_Send(&size, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
    MPI_Send(knots3.data(), size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD);
    MPI_Recv(&size, 1, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots4(size);
    MPI_Recv(knots4.data(), size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    lr->matchParametricEdge(SOUTH, knots4, true);

    knots2 = lr->getEdgeKnots(WEST, true);
    size = knots2.size();
    MPI_Send(&size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    MPI_Send(knots2.data(), size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);

  } else if(rank == 2) {
    int size;
    MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots3(size);
    MPI_Recv(knots3.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<double> knots4 = lr->getEdgeKnots(NORTH, true);
    size = knots4.size();
    MPI_Send(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Send(knots4.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    lr->matchParametricEdge(NORTH, knots3, true);
  }
#endif
}

static int check2(const vector<LRSplineSurface*> &lr) {
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


/***************** Case 3: Star  **************************
 *          ^
 *         / \
 *        /   \
 *  _____/ #2  \_____
 *  \    \     /    /
 *   \ #3 \   / #1 /
 *    \____\ /____/
 *    /    /`\    \
 *   / #4 /   \ #6 \
 *  /____/ #5  \____\
 *       \     /
 *        \   /
 *         \ /
 *          V
 *
 *********************************************************/

static vector<LRSplineSurface*> geom3() {
  return readFile("../geometries/star.g2");
}

static LRSplineSurface* mpigeom3(int rank) {
  vector<LRSplineSurface*> allPatches = geom3();
  return allPatches[rank];
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

static void mpifix3(int rank, LRSplineSurface* lr) {
#ifdef HAVE_MPI
  if (rank > 5)
    return;

  int change = 1;
  while (change) {
    change = 0;
    vector<double> knots1 = lr->getEdgeKnots(WEST, true);
    vector<double> knots2 = lr->getEdgeKnots(SOUTH, true);
    if (rank == 0) {
      // recv south from 1
      int size;
      MPI_Recv(&size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector<double> knots3(size);
      MPI_Recv(knots3.data(), size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change += lr->matchParametricEdge(WEST,  knots3, true);

      // recv west from 5
      MPI_Recv(&size, 1, MPI_INT, 5, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector<double> knots4(size);
      MPI_Recv(knots4.data(), size, MPI_DOUBLE, 5, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change += lr->matchParametricEdge(SOUTH,  knots4, true);

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
      change += lr->matchParametricEdge(WEST,  knots2, true);
      change += lr->matchParametricEdge(SOUTH,  knots1, true);
    }
    change += lr->matchParametricEdge(WEST,  knots2, true);

    int change2 = change;
    MPI_Allreduce(&change2, &change, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  vector<double> knots1 = lr->getEdgeKnots(WEST, true);
  vector<double> knots2 = lr->getEdgeKnots(SOUTH, true);
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
#endif
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

/***************** Case 4: Self-loop   ********************
 *
 *           __________,
 *          /           \
 *         /             \.
 *        /      .,       |
 *       /      |  )      |
 *       +------+-'       |
 *       |      |   #2    |
 *       |  #1  |        /`
 *       |      |       /
 *       +------+------'
 *
 *
 *********************************************************/

static vector<LRSplineSurface*> geom4() {
  return readFile("../geometries/self-loop.g2");
}

static void fix4(vector<LRSplineSurface*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    vector<double> knots1 = lr[0]->getEdgeKnots(EAST,  true);
    vector<double> knots2 = lr[0]->getEdgeKnots(NORTH, true);
    vector<double> knots3 = lr[1]->getEdgeKnots(WEST,  true);
    vector<double> knots4 = lr[1]->getEdgeKnots(EAST,  true);
    change |= lr[0]->matchParametricEdge(EAST,  knots3, true);
    change |= lr[0]->matchParametricEdge(NORTH, knots4, true);
    change |= lr[1]->matchParametricEdge(WEST,  knots1, true);
    change |= lr[1]->matchParametricEdge(EAST,  knots2, true);
  }
}

static int check4(const vector<LRSplineSurface*> &lr) {
  if(! check_matching_knots(lr[0]->getEdgeKnots(EAST, true),
                            lr[1]->getEdgeKnots(WEST, true)))
    return 1;
  if(! check_matching_knots(lr[0]->getEdgeKnots(NORTH, true),
                            lr[1]->getEdgeKnots(EAST,  true)))
    return 1;
  if(! check_isotropic_refinement(lr[0]) ||
     ! check_isotropic_refinement(lr[1]))
    return 1;
}


/***************** Legacy code   **************************
 *
 * This is kept around so I see what the basic way on how to
 * do the MPI calls.
 *
 **********************************************************/

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

/***************** MPI  methods ***************************
 *
 * Specialized functions only needed in the MPI runs
 *
 **********************************************************/

#ifdef HAVE_MPI
vector<LRSplineSurface*> collectPatches(int rank, int numProc, LRSplineSurface* patch) {
  vector<LRSplineSurface*> result;
  if(rank == 0) {
    result.push_back(patch);
    for(int i=1; i<numProc; i++) {
      int source = i;
      int tag    = i;
      int size;
      MPI_Recv(&size,      1,    MPI_INT,  source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      char serialized[size];
      MPI_Recv(serialized, size, MPI_CHAR, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      string str(serialized);
      result.push_back(new LRSplineSurface());
      stringstream(str) >> *result.back();
    }
  } else {
    stringstream ss;
    ss << *patch;
    string serialized = ss.str();
    int size = serialized.size() + 1; // sending this as char-array so end with a terminating 0

    int dest = 0;
    int tag  = rank;
    MPI_Send(&size,                 1, MPI_INT,  dest, tag, MPI_COMM_WORLD);
    MPI_Send(serialized.c_str(), size, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
  }
  return result;
}
#endif

/***************** Main method  ***************************
 *
 * The real program starts here....
 *
 **********************************************************/

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
  int result = 0;

  vector<LRSplineSurface*> model;
  int geometry   = atoi(argv[1]);
  int refinement = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int               patches[]= {      0,       0,         3,        6,        2};
  geom_function     geom[]   = {nullptr, nullptr,     geom2,    geom3,    geom4};
  mpigeom_function  mpigeom[]= {nullptr, nullptr,  mpigeom2, mpigeom3,  nullptr};
  check_function    check[]  = {nullptr, nullptr,    check2,   check3,   check4};
  fix_function      fix[]    = {nullptr, nullptr,      fix2,     fix3,     fix4};
  mpifix_function   mpifix[] = {nullptr, nullptr,   mpifix2,  mpifix3,  nullptr};
  refine_function   refine[] = {nullptr, refine1,   refine2};
  mpiref_function   mpiref[] = {nullptr, mpiref1,   mpiref2};

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << rank << ": Initialized" << endl;
  if(rank >= patches[geometry]) {
    MPI_Finalize();
    return 0;
  }

  LRSplineSurface *patch = mpigeom[geometry](rank);
  cout << "Read geometry\n";
  for(int i=0; i<iterations; i++) {
    mpiref[refinement](rank, patch);
    cout << "Did refinement - " << rank << endl;
    mpifix[geometry](  rank, patch);
    cout << "Fixed refinement - " << rank << endl;

    patch->generateIDs(); // indexing is reset after refinement
  }
  model = collectPatches(rank, patches[geometry], patch);
  cout << "Collected patches - " << rank << endl;

  if(rank == 0)
    result = check[geometry](model);
  else
    result = 0;
  MPI_Finalize();

#else
  model = geom[geometry]();

  for(int i=0; i<iterations; i++) {
    refine[refinement](model);
    fix[geometry](    model);

    for(auto lr : model)
      lr->generateIDs(); // indexing is reset after refinement
  }
  result = check[geometry](model);
#endif

  // print results to file for manual debugging
  writeFile("mesh.lr", model);
  return result;
}
