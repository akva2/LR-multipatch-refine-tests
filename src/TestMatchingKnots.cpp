#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "LRSpline/LRSplineSurface.h"
#include "GoTools/geometry/SplineSurface.h"
#include "GoTools/geometry/ObjectHeader.h"

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

  if (atoi(argv[1]) == 1)
    return case1();
  else if (atoi(argv[1]) == 2)
    return case2();

  return 1;
} 
