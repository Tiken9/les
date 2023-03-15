import sympy
import collections
from sympy.core import relational

from les.mp_model.mp_model_builder import binary_mp_variable
from les.mp_model.mp_model_builder import mp_constraint
from les.mp_model.mp_model_builder import mp_objective
from les.mp_model.mp_model_builder import mp_variable
from les.mp_model.mp_model_builder import mp_model_builder

_SYMPY_MPS_SENSE_MAPPING = {
    "<=": "L",
    ">=": "G",
    "==": "E",
}


class QuadMPBuilder(mp_model_builder.MPModelBuilder):
    def __init__(self, ):
        super().__init__()

    def convert_expr(self, expr):
        new_vars, new_constrs, new_expr = [], [], 0
        for i in expr.args:
            if i.is_Number:
                new_expr += i
            elif isinstance(i.as_coeff_Mul()[1], binary_mp_variable.BinaryMPVariable):
                    new_expr += i
            else:
                (coeff, vars) = i.as_coeff_Mul()
                var1, var2 = vars.args
                if var2 == 2:
                    new_expr += coeff*var1
                else:
                    name = var1.get_name() + '|' + var2.get_name()
                    new_vars.append(binary_mp_variable.BinaryMPVariable(name=name))
                    new_expr += coeff * new_vars[-1]
                    new_constrs.append([
                        new_vars[-1] - var1 <= 0,
                        new_vars[-1] - var2 <= 0,
                        var1 + var2 - new_vars[-1] <= 1
                    ])


        return new_vars, new_constrs, new_expr

    def quad_maximize(self, expr, name=None):
        new_vars, new_constrs, new_expr = self.convert_expr(expr)
        self.add_new_vars(new_vars, new_constrs)

        self.set_objective(new_expr, maximization=True, name=name)

    def add_new_vars(self, new_vars, new_constraints):
        for i, _var in enumerate(new_vars):
            if _var not in self._vars:
                self.add_variable(_var)

                for constr in new_constraints[i]:
                    print(self.add_constraint(constr))


    def quad_minimize(self, expr):
        new_vars, new_constrs, new_expr = self.convert_expr(expr)
        self.add_new_vars(new_vars, new_constrs)

        self.set_objective(expr, maximization=False)

    def set_quad_constraints(self, constraints):
        if not isinstance(constraints, collections.Iterable):
            raise TypeError()
        for constraint in constraints:
            fixed_expr = constraint.lhs
            new_vars, new_constrs, new_expr = self.convert_expr(fixed_expr)
            self.add_new_vars(new_vars, new_constrs)
            print(self.add_constraint(new_expr, _SYMPY_MPS_SENSE_MAPPING[constraint.rel_op], constraint.rhs))

