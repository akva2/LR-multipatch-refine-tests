#include <iostream>
#include <sstream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "LRSpline/LRSplineVolume.h"
#include "LRSpline/Meshline.h"
#include "GoTools/trivariate/SplineVolume.h"
#include "GoTools/geometry/ObjectHeader.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define DOUBLE_TOL 1e-5

using namespace std;
using namespace LR;
using namespace Go;

// serial checking of global model
typedef int  (*check_function)( const vector<LRSplineVolume*>&);

// refinement, geometry and topology functions related to each case (SERIAL)
typedef void                    (*refine_function)(const vector<LRSplineVolume*>&);
typedef void                    (*fix_function)(         vector<LRSplineVolume*>&);
typedef vector<LRSplineVolume*> (*geom_function)();

// refinement, geometry and topology functions related to each case (MPI)
typedef void            (*mpiref_function)( int, LRSplineVolume*);
typedef void            (*mpifix_function)( int, LRSplineVolume*);
typedef LRSplineVolume* (*mpigeom_function)(int);

/***************** MPI  methods ***************************
 *
 * Specialized functions only needed in the MPI runs
 *
 **********************************************************/

#ifdef HAVE_MPI

static void reverse_u(const vector<Meshline*>& knots);
static void reverse_v(const vector<Meshline*>& knots);
static void flip_uv(  const vector<Meshline*>& knots);

vector<LRSplineVolume*> collectPatches(int rank, int numProc, LRSplineVolume* patch) {
  vector<LRSplineVolume*> result;
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
      result.push_back(new LRSplineVolume());
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

void sendLines(int rank, int tag, const LRSplineVolume* lr, parameterEdge edg, int orient=0) {
  // fetch local knots
  vector<Meshline*> knots = lr->getEdgeKnots(edg, true);
  // orient if needed
  if(orient & 1)
    reverse_v(knots);
  if(orient & 2)
    reverse_u(knots);
  if(orient & 4)
    flip_uv(knots);
  // wrap in simple containers
  int size = knots.size();
  vector<double> constPar, start, stop;
  vector<int>    mult, span_u;
  for(auto m : knots) {
    span_u.push_back(  m->span_u_line_);
    constPar.push_back(m->const_par_);
    start.push_back(   m->start_);
    stop.push_back(    m->stop_);
    mult.push_back(    m->multiplicity_);
  }
  // send it all across the network
  MPI_Send(&size,              1, MPI_INT,    rank, tag++, MPI_COMM_WORLD);
  MPI_Send(span_u.data(),   size, MPI_INT,    rank, tag++, MPI_COMM_WORLD);
  MPI_Send(constPar.data(), size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD);
  MPI_Send(start.data(),    size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD);
  MPI_Send(stop.data(),     size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD);
  MPI_Send(mult.data(),     size, MPI_INT,    rank, tag++, MPI_COMM_WORLD);

  // clean up
  for(auto m : knots)
    delete m;
}

vector<Meshline*> recvLines(int rank, int tag) {
  // fetch number of meshlines
  int size;
  MPI_Recv(&size, 1, MPI_INT,    rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  vector<double> constPar(size), start(size), stop(size);
  vector<int>    mult(size), span_u(size);
  vector<Meshline*> results;

  // send it all across the network
  MPI_Recv(span_u.data(),   size, MPI_INT,    rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(constPar.data(), size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(start.data(),    size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(stop.data(),     size, MPI_DOUBLE, rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(mult.data(),     size, MPI_INT,    rank, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(int i=0; i<size; i++)
    results.push_back(new Meshline(span_u[i], constPar[i], start[i], stop[i], mult[i]));
  return results;

}
#endif

// read multipatch g2 file and return LRSplineVolume representation of all patches
vector<LRSplineVolume*> readFile(const string &filename) {
  vector<LRSplineVolume*> results;
  ObjectHeader head;
  SplineVolume sv;

  ifstream in(filename);
  if(!in.good()) {
    cerr << "could not open file: " << filename << endl;
    return results;
  }

  while(!in.eof()) { // for all patches
    in >> head >> sv;
    results.push_back(new LRSplineVolume(&sv));
    ws(in); // eat whitespaces
  }

  for(auto lr : results)
    lr->generateIDs();
  return results;
}

// write multipatch lr model to file for manual inspection
void writeFile(const string &filename, vector<LRSplineVolume*> model) {
  if(model.size() == 0) return;
  ofstream out(filename);
  for(auto lr : model)
    out << *lr << endl;
}

/***************** TOPOLOGY  FUNCTIONS  ******************
 *
 * Fixes local orientation issues. Swaps parametric directions and reverses ranges
 *
 *********************************************************/

static void reverse_u(const vector<Meshline*>& knots) {
  for(auto l : knots) {
    if(l->is_spanning_u()) {
      double tmp = l->start_;
      l->start_ = 1.0 - l->stop_;
      l->stop_  = 1.0 - tmp;
    } else {
      l->const_par_ = 1.0 - l->const_par_;
    }
  }
}

static void reverse_v(const vector<Meshline*>& knots) {
  for(auto l : knots) {
    if(l->is_spanning_u()) {
      l->const_par_ = 1.0 - l->const_par_;
    } else {
      double tmp = l->start_;
      l->start_ = 1.0 - l->stop_;
      l->stop_  = 1.0 - tmp;
    }
  }
}

static void flip_uv(const vector<Meshline*>& knots) {
  for(auto l : knots)
    l->span_u_line_ = !l->span_u_line_;
}

/***************** ASSERTION FUNCTIONS  ******************
 *
 * logical tests to see if the resulting multipatch model
 * is indeed matching.
 *
 *********************************************************/

// check that all meshlines from a are contained in b (call this twice and
// swap the roles of a & b when testing for assertions). Note: boundary meshlines
// may not equally match from both sides (multiplicites and overlaps on lines may occur)
// and this is the reason we need to see if all lines from one side is contained
// in the lines from the other side
static bool check_contained_in(const vector<Meshline*> a, const vector<Meshline*> b) {
  for(auto l1 : a) {
    bool found = false;
    for(auto l2 : b) {
      if(l1->is_spanning_u() == l2->is_spanning_u()) {
        if(fabs(l1->const_par_ - l2->const_par_) < DOUBLE_TOL &&
           l1->start_ >= l2->start_ &&
           l1->stop_  <= l2->stop_) {
          found = true;
          break;
        }
      }
    }
    if(!found)
      return false;
  }

  return true;
}

// check that all elements continue to be perfect squares (isotropic / basis-function refinement)
static bool check_isotropic_refinement(LRSplineVolume *lr) {
  for(auto el : lr->getAllElements())
    if(fabs( (el->umax() - el->umin()) - (el->vmax() - el->vmin()) ) >1e-4 ||
       fabs( (el->umax() - el->umin()) - (el->wmax() - el->wmin()) ) >1e-4)
      return false;

  return true;
}

/***************** REFINEMENT FUNCTIONS  *****************
 *
 * different refinement schemes that is used to test all
 * geometries.
 *
 *********************************************************/

// Refinement strategy 1: refine all 8 corners of the FIRST patch
static void refine1(const vector<LRSplineVolume*>& model) {
  vector<Basisfunction*> oneFunction;
  vector<int> corner_idx;
  // fetch index of all 8 corners
  for(int i=0; i<2; i++) {
    for(int j=0; j<2; j++) {
      for(int k=0; k<2; k++) {
        int edg = 0;
        edg |= (i) ? WEST  : EAST;
        edg |= (j) ? NORTH : SOUTH;
        edg |= (k) ? TOP   : BOTTOM;
        model[0]->getEdgeFunctions(oneFunction, (parameterEdge) edg);
        corner_idx.push_back(oneFunction[0]->getId());
      }
    }
  }
  model[0]->refineBasisFunction(corner_idx);
}

// Refinement strategy 1: MPI implementation
static void mpiref1(int rank, LRSplineVolume *patch) {
  if(rank > 0) return; // skip all but the first patch
  refine1(vector<LRSplineVolume*>(1, patch));
}

// Refinement strategy 2: randomly one function on all patches independently
static void refine2(const vector<LRSplineVolume*>& model) {
  for(size_t i=0; i<model.size(); ++i)
    model[i]->refineBasisFunction((13*i+17) % model[i]->nBasisFunctions());
}

// Refinement strategy 2: MPI implementation
static void mpiref2(int rank, LRSplineVolume *patch) {
  patch->refineBasisFunction((13*rank+17) % patch->nBasisFunctions());
}

// Refinement strategy 3: refine SOUTH-WEST-BOTTOM corner of the FIRST patch
static void refine3(const vector<LRSplineVolume*>& model) {
  vector<Basisfunction*> oneFunction;
  vector<int> corner_idx;
  // fetch index of all 4 corners
  model[0]->getEdgeFunctions(oneFunction, (parameterEdge) SOUTH|WEST|TOP);
  corner_idx.push_back(oneFunction[0]->getId());
  // do the actual refinement
  model[0]->refineBasisFunction(corner_idx);
}

// Refinement strategy 3: MPI implementation
static void mpiref3(int rank, LRSplineVolume *patch) {
  if(rank > 0) return; // skip all but the first patch
  refine3(vector<LRSplineVolume*>(1, patch));
}

/***************** Case 1: L-shape  **********************
 *
 * |z-axis
 * +--------+
 * |        |
 * |    #3  |           (extruded into y-plane)
 * |        |
 * +--------+--------+
 * |        |        |
 * |   #1   |   #2   |
 * |        |        |
 * +--------+--------+--> x-axis
 *
 *********************************************************/

static vector<LRSplineVolume*> geom1() {
  return readFile("../geometries/3D/boxes-3.g2");
}

static LRSplineVolume* mpigeom1(int rank) {
  vector<LRSplineVolume*> allPatches = geom1();
  return allPatches[rank]->copy();
}

static void fix1(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(EAST, true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(WEST, true);
    change |= lr[0]->matchParametricEdge(EAST, knots2, true);
    change |= lr[1]->matchParametricEdge(WEST, knots1, true);

    // match patch 1 to patch 2
    vector<Meshline*> knots3 = lr[0]->getEdgeKnots(TOP,    true);
    vector<Meshline*> knots4 = lr[2]->getEdgeKnots(BOTTOM, true);
    change |= lr[0]->matchParametricEdge(TOP,    knots4, true);
    change |= lr[2]->matchParametricEdge(BOTTOM, knots3, true);
  }
}

static void mpifix1(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, EAST, 0);
      sendLines(2, 1, lr, TOP,  0);
      vector<Meshline*> knots1 = recvLines(1, 1);
      vector<Meshline*> knots2 = recvLines(2, 1);
      change |= lr->matchParametricEdge(EAST, knots1, true);
      change |= lr->matchParametricEdge(TOP,  knots2, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  2, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Recv(&others_change, 1, MPI_INT,  2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, WEST, 0);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  2, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 2) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, BOTTOM, 0);
      change |= lr->matchParametricEdge(BOTTOM, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check1(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(EAST, true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(WEST, true);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  vector<Meshline*> knots3 = lr[0]->getEdgeKnots(TOP,    true);
  vector<Meshline*> knots4 = lr[2]->getEdgeKnots(BOTTOM, true);
  if(!check_contained_in(knots3, knots4)) return 1;
  if(!check_contained_in(knots4, knots3)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}

/***************** Case 2: Disc stair  *******************
 *
 * z-axis
 *          +--------+
 * ^        |        |
 * |        |   #3   |  (extruded in r=(1,2))
 * |        |        |
 * +--------+--------+
 * |        |        |
 * |   #1   |   #2   |
 * |        |        |
 * +--------+--------+--> theta-axis
 *
 *********************************************************/

static vector<LRSplineVolume*> geom2() {
  return readFile("../geometries/3D/disc-stair-3.g2");
}

static LRSplineVolume* mpigeom2(int rank) {
  vector<LRSplineVolume*> allPatches = geom2();
  return allPatches[rank]->copy();
}

static void fix2(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(NORTH, true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(SOUTH, true);
    change |= lr[0]->matchParametricEdge(NORTH, knots2, true);
    change |= lr[1]->matchParametricEdge(SOUTH, knots1, true);

    // match patch 1 to patch 2
    vector<Meshline*> knots3 = lr[1]->getEdgeKnots(TOP,    true);
    vector<Meshline*> knots4 = lr[2]->getEdgeKnots(BOTTOM, true);
    change |= lr[1]->matchParametricEdge(TOP,    knots4, true);
    change |= lr[2]->matchParametricEdge(BOTTOM, knots3, true);
  }
}

static void mpifix2(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, NORTH, 0);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(NORTH, knots, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  2, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Recv(&others_change, 1, MPI_INT,  2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots1 = recvLines(0, 1);
      vector<Meshline*> knots2 = recvLines(2, 1);
      sendLines(0, 1, lr, SOUTH, 0);
      sendLines(2, 1, lr, TOP,   0);
      change |= lr->matchParametricEdge(SOUTH, knots1, true);
      change |= lr->matchParametricEdge(TOP,   knots2, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  2, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 2) {
      sendLines(1, 1, lr, BOTTOM, 0);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(BOTTOM, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check2(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(NORTH, true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(SOUTH, true);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  vector<Meshline*> knots3 = lr[1]->getEdgeKnots(TOP,    true);
  vector<Meshline*> knots4 = lr[2]->getEdgeKnots(BOTTOM, true);
  if(!check_contained_in(knots3, knots4)) return 1;
  if(!check_contained_in(knots4, knots3)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}

/***************** Case 3: Orient 1    *******************
 *
 *          z    y
 *          ^   /                Patch 1        Patch 2
 *    ______|__/________,         WEST           BOTTOM
 *   /      | /        /|       +-------+     +-------+
 *  /       |/        / |       |       |     |       |
 * +--------+--------+  |     ^ |       |     |       | ^
 * |        |        |  |    w| |       |     |       | |v
 * |   #2   |   #1   | /      | +-------+     +-------+ |
 * |        |        |/       o--->                <----o
 * +--------+--------+--> x      v                   u
 *
 *********************************************************/

static vector<LRSplineVolume*> geom3() {
  return readFile("../geometries/3D/boxes-2-orient1.g2");
}

static LRSplineVolume* mpigeom3(int rank) {
  vector<LRSplineVolume*> allPatches = geom3();
  return allPatches[rank]->copy();
}

static void fix3(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST,   true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(BOTTOM, true);
    reverse_v(knots1);
    reverse_v(knots2);
    change |= lr[0]->matchParametricEdge(WEST,   knots2, true);
    change |= lr[1]->matchParametricEdge(BOTTOM, knots1, true);
  }
}

static void mpifix3(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, WEST, 1);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, BOTTOM, 1);
      change |= lr->matchParametricEdge(BOTTOM, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check3(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST,   true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(BOTTOM, true);
  reverse_v(knots2);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}

/***************** Case 4: Orient 2    *******************
 *
 *          z    y                              Patch 2
 *          ^   /                Patch 1        BOTTOM
 *    ______|__/________,         WEST      o---> u
 *   /      | /        /|       +-------+   | +-------+
 *  /       |/        / |       |       |  v| |       |
 * +--------+--------+  |     ^ |       |   v |       |
 * |        |        |  |    w| |       |     |       |
 * |   #2   |   #1   | /      | +-------+     +-------+
 * |        |        |/       o--->
 * +--------+--------+--> x      v
 *
 *********************************************************/

static vector<LRSplineVolume*> geom4() {
  return readFile("../geometries/3D/boxes-2-orient2.g2");
}

static LRSplineVolume* mpigeom4(int rank) {
  vector<LRSplineVolume*> allPatches = geom4();
  return allPatches[rank]->copy();
}

static void fix4(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST,   true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(BOTTOM, true);
    reverse_u(knots1);
    reverse_u(knots2);
    change |= lr[0]->matchParametricEdge(WEST,   knots2, true);
    change |= lr[1]->matchParametricEdge(BOTTOM, knots1, true);
  }
}

static void mpifix4(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, WEST, 2);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, BOTTOM, 2);
      change |= lr->matchParametricEdge(BOTTOM, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check4(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST,   true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(BOTTOM, true);
  reverse_u(knots2);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}

/***************** Case 5: Orient 4    *******************
 *
 *          z    y
 *          ^   /                Patch 1        Patch 2
 *    ______|__/________,         WEST           WEST
 *   /      | /        /|       +-------+     +-------+
 *  /       |/        / |       |       |     |       |
 * +--------+--------+  |     ^ |       |   ^ |       |
 * |        |        |  |    w| |       |  v| |       |
 * |   #2   |   #1   | /      | +-------+   | +-------+
 * |        |        |/       o--->         o---->
 * +--------+--------+--> x      v               w
 *
 *********************************************************/

static vector<LRSplineVolume*> geom5() {
  return readFile("../geometries/3D/boxes-2-orient4.g2");
}

static LRSplineVolume* mpigeom5(int rank) {
  vector<LRSplineVolume*> allPatches = geom5();
  return allPatches[rank]->copy();
}

static void fix5(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST, true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(WEST, true);
    flip_uv(knots1);
    flip_uv(knots2);
    change |= lr[0]->matchParametricEdge(WEST, knots2, true);
    change |= lr[1]->matchParametricEdge(WEST, knots1, true);
  }
}

static void mpifix5(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, WEST, 4);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, WEST, 4);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check5(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST, true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(WEST, true);
  flip_uv(knots2);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}

/***************** Case 6: Orient 6    *******************
 *
 *          z    y                              Patch 2
 *          ^   /                Patch 1         TOP
 *    ______|__/________,         WEST      o---> v
 *   /      | /        /|       +-------+   | +-------+
 *  /       |/        / |       |       |  u| |       |
 * +--------+--------+  |     ^ |       |   v |       |
 * |        |        |  |    w| |       |     |       |
 * |   #2   |   #1   | /      | +-------+     +-------+
 * |        |        |/       o--->
 * +--------+--------+--> x      v
 *
 *********************************************************/

static vector<LRSplineVolume*> geom6() {
  return readFile("../geometries/3D/boxes-2-orient6.g2");
}

static LRSplineVolume* mpigeom6(int rank) {
  vector<LRSplineVolume*> allPatches = geom6();
  return allPatches[rank]->copy();
}

static void fix6(vector<LRSplineVolume*> &lr) {
  bool change = true;
  while(change) {
    change = false;
    // match patch 0 to patch 1
    vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST, true);
    vector<Meshline*> knots2 = lr[1]->getEdgeKnots(TOP, true);
    reverse_u(knots1);
    flip_uv(knots1);
    reverse_u(knots2);
    flip_uv(knots2);
    change |= lr[0]->matchParametricEdge(WEST, knots2, true);
    change |= lr[1]->matchParametricEdge(TOP,  knots1, true);
  }
}

static void mpifix6(int rank, LRSplineVolume* lr) {
#ifdef HAVE_MPI
  int change = 1;
  while(change) {
    change = 0;
    int others_change = 0;

    if(rank == 0) {
      sendLines(1, 1, lr, WEST, 6);
      vector<Meshline*> knots = recvLines(1, 1);
      change |= lr->matchParametricEdge(WEST, knots, true);

      MPI_Send(&change,        1, MPI_INT,  1, 10, MPI_COMM_WORLD);
      MPI_Recv(&others_change, 1, MPI_INT,  1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
    } else if(rank == 1) {
      vector<Meshline*> knots = recvLines(0, 1);
      sendLines(0, 1, lr, TOP, 6);
      change |= lr->matchParametricEdge(TOP, knots, true);

      MPI_Recv(&others_change, 1, MPI_INT,  0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      change |= others_change;
      MPI_Send(&change,        1, MPI_INT,  0, 10, MPI_COMM_WORLD);
    }
  }
#endif
}

static int check6(const vector<LRSplineVolume*> &lr) {
  vector<Meshline*> knots1 = lr[0]->getEdgeKnots(WEST, true);
  vector<Meshline*> knots2 = lr[1]->getEdgeKnots(TOP, true);
  reverse_u(knots2);
  flip_uv(knots2);
  if(!check_contained_in(knots1, knots2)) return 1;
  if(!check_contained_in(knots2, knots1)) return 1;

  for(auto l : lr)
    if(! check_isotropic_refinement(l) )
      return 1;
  return 0;
}


/***************** Main method  ***************************
 *
 * The real program starts here....
 *
 **********************************************************/

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "File usage: " << argv[0] << " <geometry> <refinement> <iterations>" << std::endl;
    std::cerr << "  geometry: " << std::endl;
    std::cerr << "    1 = L-shape, 3 patches" << std::endl;
    std::cerr << "    2 = Disc-stair, 3 patches" << std::endl;
    std::cerr << "    3 = Orientation=1, 2 patches" << std::endl;
    std::cerr << "    4 = Orientation=2, 2 patches" << std::endl;
    std::cerr << "    5 = Orientation=4, 2 patches" << std::endl;
    std::cerr << "    6 = Orientation=6, 2 patches" << std::endl;
    std::cerr << "  refinement: " << std::endl;
    std::cerr << "    1 = All corners on first patch" << std::endl;
    std::cerr << "    2 = One random function on each patch (chosen independently)" << std::endl;
    std::cerr << "    3 = South-East-Bottom corner on first patch" << std::endl;
    std::cerr << "  iterations: (int) number of refinement that is to be performed" << std::endl;
    return 1;
  }
  int result = 0;

  vector<LRSplineVolume*> model;
  int geometry   = atoi(argv[1]);
  int refinement = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int               patches[]= {      0,        3,        3,        2,        2,        2,        2};
  geom_function     geom[]   = {nullptr,    geom1,    geom2,    geom3,    geom4,    geom5,    geom6};
  mpigeom_function  mpigeom[]= {nullptr, mpigeom1, mpigeom2, mpigeom3, mpigeom4, mpigeom5, mpigeom6};
  check_function    check[]  = {nullptr,   check1,   check2,   check3,   check4,   check5,   check6};
  fix_function      fix[]    = {nullptr,     fix1,     fix2,     fix3,     fix4,     fix5,     fix6};
  mpifix_function   mpifix[] = {nullptr,  mpifix1,  mpifix2,  mpifix3,  mpifix4,  mpifix5,  mpifix6};
  refine_function   refine[] = {nullptr,  refine1,  refine2,  refine3};
  mpiref_function   mpiref[] = {nullptr,  mpiref1,  mpiref2,  mpiref3};

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << rank << ": Initialized" << endl;
  if(rank >= patches[geometry]) {
    MPI_Finalize();
    return 0;
  }

  LRSplineVolume *patch = mpigeom[geometry](rank);
  cout << rank << ": " << "Read geometry" << endl;
  for(int i=0; i<iterations; i++) {
    mpiref[refinement](rank, patch);
    cout << rank << ": " << "Did refinement" << endl;
    mpifix[geometry](  rank, patch);
    cout << rank << ": " << "Fixed refinement" << endl;

    patch->generateIDs(); // indexing is reset after refinement
  }
  model = collectPatches(rank, patches[geometry], patch);
  cout << rank << ": " << "Collected patches" << endl;

  if(rank == 0)
    result = check[geometry](model);
  else
    result = 0;
  cout << rank << ": " << "Closing..." << endl;
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
