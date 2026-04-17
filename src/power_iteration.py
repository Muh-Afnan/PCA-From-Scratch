from matrix_library.src.matrix import Matrix
import math

class PowerIteration():
    def _norm(self,v)->"int | float":
        return math.sqrt(sum(x*x for x in v))
    

    def calculate_eigen(self,matrix:Matrix,iterations=1000, tol=1e-6):
        n = len(matrix.data)

        eigen_value = 0
        v = [1]*n

        for i in range(iterations):
            v = Matrix(v)
            AV = matrix * v
            norm = self._norm(AV)
            v_new = [x / norm for x in AV]

            if sum(abs(v_new[i]-v.data[i]) for i in range(n))<tol:
                v.data = v_new
                break
            v.data = v_new

        AV = matrix * v
        eigen_value = AV @ v
        return eigen_value,v

A = Matrix([[1, 2], [3, 4]])
scalar_obj = PowerIteration()
result = scalar_obj.calculate_eigen(A)

print(result)