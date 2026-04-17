from matrix_library.src.matrix import Matrix
import math

class StandardScaler():

    def mean(self, data:list)->int|float:
        return (sum(data)/len(data))
    
    def std(self, col_mean,data:list)->int|float:
         return math.sqrt(sum((x-col_mean)**2 for x in data)/(len(data)-1))
    
    def scalar(self,data:Matrix)->Matrix:
        raw = data.data
        trans_matrix = list(zip(*raw))
        scaled = []
        for col in trans_matrix:
            col_mean = self.mean(col)
            col_std = self.std(col_mean,col)
            normalized =[(x-col_mean)/col_std for x in col]
            scaled.append(normalized)
        return Matrix([list(row) for row in zip(*scaled)])
    

A = Matrix([[1, 2], [3, 4]])
scalar_obj = StandardScaler()
result = scalar_obj.scalar(A)

print(result)