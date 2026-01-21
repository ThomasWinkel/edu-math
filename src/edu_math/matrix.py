from __future__ import annotations
import random

ScalarTypes = int | float | complex
VectorType = list[ScalarTypes]
MatrixType = list[VectorType]
InputTypes = ScalarTypes | VectorType | MatrixType


class Matrix:
    data: MatrixType

    def __init__(self, data: InputTypes = None) -> None:
        if data is None:
            self.data = [[]]
        elif isinstance(data, ScalarTypes):
            self.data = [[data]]
        elif isinstance(data, list):
            if len(data) == 0:
                self.data = [[]]
            elif isinstance(data[0], ScalarTypes):
                # Fall: 1D Liste (Vektor) -> Wird als 1xN Matrix (Zeilenvektor) behandelt
                if not all(isinstance(x, ScalarTypes) for x in data):
                     raise TypeError("Matrix darf nur Zahlen enthalten.")
                self.data = [data]
            elif isinstance(data[0], list):
                #Validierung: Alle Zeilen müssen Listen sein und die gleiche Länge haben
                row_len = len(data[0])
                for row in data:
                    if not isinstance(row, list):
                        raise ValueError("Gemischte Dimensionen: Liste enthält Listen und Skalare.")
                    if len(row) != row_len:
                        raise ValueError("Nicht-rechteckige Matrix: Zeilen haben unterschiedliche Längen.")
                    # Optional: Prüfen ob Elemente Zahlen sind
                    if not all(isinstance(x, ScalarTypes) for x in row):
                        raise TypeError("Matrix darf nur Zahlen enthalten.")
                self.data = data
            else:
                raise ValueError("Invalid input.")
        else:
            raise ValueError("Invalid input.")
    
    def __getitem__(self, index):
        # Fall 1: Zugriff auf ein Element oder Slice via matrix[row, col]
        if isinstance(index, tuple):
            row_idx, col_idx = index
            
            # Zeilen extrahieren
            if isinstance(row_idx, slice):
                selected_rows = self.data[row_idx]
            else:
                selected_rows = [self.data[row_idx]]
            
            # Spalten aus den ausgewählten Zeilen extrahieren
            result_data = []
            for row in selected_rows:
                if isinstance(col_idx, slice):
                    result_data.append(row[col_idx])
                else:
                    result_data.append([row[col_idx]])
            
            # Wenn das Ergebnis nur ein einzelner Wert ist, gib ihn direkt zurück
            if not isinstance(row_idx, slice) and not isinstance(col_idx, slice):
                return self.data[row_idx][col_idx]
            
            # Ansonsten gib eine neue Matrix zurück (Slicing-Ergebnis)
            return Matrix(result_data)
        
        # Fall 2: Zugriff auf ganze Zeile via matrix[row]
        return self.data[index]

    def __str__(self):
        #return '\n'.join(['\t'.join(map(str, row)) for row in self.data])  10.4g
        return "\n".join(["\t".join([f"{val:10.3g}" for val in row]) for row in self.data])
    
    def __repr__(self):
        if not self.data or not self.data[0]:
            return "Matrix(empty)"
        return f"Matrix({self.data}) [{len(self.data)}x{len(self.data[0])}]"
    
    def __mul__(self, other) -> Matrix:
        # Skalarmultiplikation
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            result = [[other * value for value in row] for row in self.data]
            return Matrix(result)
        
        # Matrizenmultiplikation
        if isinstance(other, Matrix):
            if len(self.data) != len(other.data) & len(self.data[0]) != len(other.data[0]):
                raise ValueError("Incompatible matrix sizes.")
            
            result = [[self.data[m][n] * other.data[m][n] for n in range(len(self.data[m]))] for m in range(len(self.data))]

            return Matrix(result)
        
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other) -> Matrix:
        # Matrizenmultiplikation
        if isinstance(other, Matrix):
            if len(self.data[0]) != len(other.data):
                raise ValueError("Incompatible matrix sizes.")
            
            result = []
            for i in range(len(self.data)):
                row = []
                for k in range(len(other.data[0])):
                    current_sum = 0
                    for j in range(len(self.data[0])):
                        current_sum += self.data[i][j] * other.data[j][k]
                    row.append(current_sum)
                result.append(row)
            return Matrix(result)
        
        return NotImplemented

    def __rmatmul__(self, other):
        return self.__matmul__(other)
    
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Incompatible type.")
        
        if len(self.data) != len(other.data) or len(self.data[0]) != len(other.data[0]):
            raise ValueError("Incompatible matrix sizes.")
            
        result = [[self.data[m][n] + other.data[m][n] for n in range(len(self.data[m]))] for m in range(len(self.data))]
        return Matrix(result)
    
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Incompatible type.")
        
        if len(self.data) != len(other.data) or len(self.data[0]) != len(other.data[0]):
            raise ValueError("Incompatible matrix sizes.")
            
        result = [[self.data[m][n] - other.data[m][n] for n in range(len(self.data[m]))] for m in range(len(self.data))]
        return Matrix(result)
    
    @staticmethod
    def identity(size):
        data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        return Matrix(data)
    
    @staticmethod
    def ones(n = 1, m = 1):
        data = [[1 for j in range(n)] for i in range(m)]
        return Matrix(data)
    
    @staticmethod
    def rand(n = 1, m = 1):
        data = [[random.random() for j in range(n)] for i in range(m)]
        return Matrix(data)
    
    @staticmethod
    def randn(n = 1, m = 1, mu = 0, sigma = 1):
        data = [[random.gauss(mu, sigma) for j in range(n)] for i in range(m)]
        return Matrix(data)
    
    @staticmethod
    def randi(n = 1, m = 1, a = 0, b = 9):
        data = [[random.randint(a, b) for j in range(n)] for i in range(m)]
        return Matrix(data)
    
    def m(self):
        return len(self.data) if len(self.data[0]) > 0 else 0
    
    def n(self):
        return len(self.data[0])
    
    def is_square(self):
        return len(self.data) == len(self.data[0])

    def get_minor(self, row, col):
        new_data = [
            [self.data[i][j] for j in range(len(self.data[0])) if j != col]
            for i in range(len(self.data)) if i != row
        ]
        return Matrix(new_data)

    def determinant(self):
        if not self.is_square():
            raise ValueError("Die Determinante ist nur für quadratische Matrizen definiert.")
        
        n = len(self.data)
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        det = 0
        for j in range(n):
            sign = (-1) ** j
            det += sign * self.data[0][j] * self.get_minor(0, j).determinant()
        
        return det
    
    def transpose(self):
        result = [[self.data[m][n] for m in range(len(self.data))] for n in range(len(self.data[0]))]
        return Matrix(result)
    
    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix ist singulär und besitzt keine Inverse (Determinante ist 0).")
        
        n = len(self.data)
        
        # Spezialfall 1x1
        if n == 1:
            return Matrix([[1 / det]])
        
        # Spezialfall 2x2 (optional, aber effizienter)
        if n == 2:
            a, b = self.data[0][0], self.data[0][1]
            c, d = self.data[1][0], self.data[1][1]
            return Matrix([[d/det, -b/det], [-c/det, a/det]])

        # Allgemeiner Fall über Kofaktoren
        cofactors = []
        for i in range(n):
            row = []
            for j in range(n):
                minor = self.get_minor(i, j)
                # Vorzeichen gemäß Schachbrettmuster
                sign = (-1) ** (i + j)
                row.append(sign * minor.determinant())
            cofactors.append(row)
        
        # Die Inverse ist (1/det) * Transponierte der Kofaktormatrix
        cofactor_matrix = Matrix(cofactors)
        adjugate = cofactor_matrix.transpose()
        
        return adjugate * (1 / det)
    
    def row_echelon_form(self):
        # Kopie der Daten erstellen, um das Original nicht zu verändern
        matrix = [row[:] for row in self.data]
        rows = len(matrix)
        cols = len(matrix[0])
        
        pivot_row = 0
        for j in range(cols):
            if pivot_row >= rows:
                break
            
            # 1. Pivot-Suche: Finde die Zeile mit dem größten Wert in Spalte j (Teil-Pivotisierung)
            max_row = pivot_row
            for i in range(pivot_row + 1, rows):
                if abs(matrix[i][j]) > abs(matrix[max_row][j]):
                    max_row = i
            
            # Falls die ganze Spalte ab hier 0 ist, überspringe sie
            if abs(matrix[max_row][j]) < 1e-10: # Kleine Zahl statt 0 wegen Rundungsfehlern
                continue
            
            # 2. Zeilen tauschen
            matrix[pivot_row], matrix[max_row] = matrix[max_row], matrix[pivot_row]
            
            # 3. Elimination der Werte unter dem Pivot
            for i in range(pivot_row + 1, rows):
                factor = matrix[i][j] / matrix[pivot_row][j]
                for k in range(j, cols):
                    matrix[i][k] -= factor * matrix[pivot_row][k]
            
            pivot_row += 1
            
        return Matrix(matrix)

    def rank(self):
        # Erst in Stufenform bringen
        ref_matrix = self.row_echelon_form()
        rank = 0
        for row in ref_matrix.data:
            # Eine Zeile zählt zum Rang, wenn mindestens ein Element != 0 ist
            if any(abs(value) > 1e-10 for value in row):
                rank += 1
        return rank
    
    @staticmethod
    def solve(a_matrix, b_vector):
        """
        Löst das System A * x = b.
        b_vector kann eine Liste oder eine Matrix mit einer Spalte sein.
        """
        # Konvertiere b_vector in eine Liste, falls es eine Matrix ist
        if isinstance(b_vector, Matrix):
            b = [row[0] for row in b_vector.data]
        else:
            b = list(b_vector)

        n = len(a_matrix.data)
        if len(b) != n:
            raise ValueError("Die Dimension von b muss der Zeilenanzahl von A entsprechen.")
        if not a_matrix._is_square():
            raise ValueError("Lösung aktuell nur für quadratische Matrizen implementiert.")

        # Erstelle die erweiterte Matrix (A|b)
        aug = [a_matrix.data[i] + [b[i]] for i in range(n)]
        
        # Gauß-Jordan-Elimination
        for i in range(n):
            # 1. Pivot-Suche (Teil-Pivotisierung)
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            if abs(aug[i][i]) < 1e-10:
                raise ValueError("System hat keine eindeutige Lösung (Determinante ist 0).")

            # 2. Pivot auf 1 normieren
            pivot = aug[i][i]
            for j in range(i, n + 1):
                aug[i][j] /= pivot

            # 3. Alle anderen Elemente in dieser Spalte eliminieren (oben und unten!)
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(i, n + 1):
                        aug[k][j] -= factor * aug[i][j]

        # Die letzte Spalte der erweiterten Matrix ist der Lösungsvektor
        #solution = [row[-1] for row in aug]
        #return solution
        return Matrix(aug)
