#include <iostream>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include "test_vars_constr_cost.h"

int main(int argc, char** argv)
{
  // 1. define the problem
  ifopt::Problem nlp;
  std::shared_ptr<ifopt::ExVariables> vars = std::make_shared<ifopt::ExVariables>();
  std::shared_ptr<ifopt::ExConstraint> cons = std::make_shared<ifopt::ExConstraint>();
  std::shared_ptr<ifopt::ExCost> cost = std::make_shared<ifopt::ExCost>();

  nlp.AddVariableSet  (vars);
  nlp.AddConstraintSet(cons);
  nlp.AddCostSet      (cost);
  nlp.PrintCurrent();

  // 2. choose solver and options
  ifopt::IpoptSolver ipopt;
  ipopt.SetOption("linear_solver", "mumps");
  ipopt.SetOption("jacobian_approximation", "exact");

  // 3 . solve
  ipopt.Solve(nlp);
  Eigen::VectorXd x = nlp.GetOptVariables()->GetValues();
  std::cout << "x = [" << x.transpose() << "]" << std::endl;

  /*Eigen::Vector2d x2;
  x2 << 0, 1;
  vars->SetVariables(x2);
  ipopt.Solve(nlp);*/

  return 0;
}