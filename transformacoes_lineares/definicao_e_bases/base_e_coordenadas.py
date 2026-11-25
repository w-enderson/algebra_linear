import numpy as np

# =======================
# Espaços vetoriais R^n
# =======================

class Base:
    """
    Representa uma base ordenada B = {b_1, ..., b_n} de um espaço vetorial R^n.
    """

    def __init__(self, vetores: list[list[float] | np.ndarray]):
        if not vetores:
            raise ValueError("A base deve conter pelo menos um vetor.")

        self.vetores = [np.asarray(b, dtype=float) for b in vetores]
        self.dim = len(self.vetores)

        # Verifica se todos os vetores pertencem ao mesmo R^n
        for b in self.vetores:
            if b.ndim != 1 or b.shape[0] != self.dim:
                raise ValueError(
                    f"Para formar uma base de R^{self.dim}, todos os {self.dim} vetores "
                    f"devem ter exatamente {self.dim} componentes."
                )

        # Verifica independência linear (o posto deve ser igual à dimensão n)
        self._B = np.column_stack(self.vetores)
        if np.linalg.matrix_rank(self._B) < self.dim:
            raise ValueError(
                "Os vetores fornecidos são linearmente dependentes (LD) e não formam uma base."
            )

    def coordenadas(self, v: list[float] | np.ndarray) -> np.ndarray:
        """
        Encontra os únicos escalares (c_1, ..., c_n) tais que v = sum(c_i * b_i).
        (a unicidade é garantida, pois os vetores b_i são LI)
        
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (self.dim,):
            raise ValueError(f"O vetor v deve pertencer a R^{self.dim}.")
        return np.linalg.solve(self._B, v)

    def reconstruir(self, coords: list[float] | np.ndarray) -> np.ndarray:
        """
        Calcula a combinação linear sum(c_i * b_i).
        """
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (self.dim,):
            raise ValueError(
                f"São necessários exatamente {self.dim} escalares para reconstruir "
                f"um vetor nesta base, mas foram fornecidos {coords.size}."
            )

        resultado = np.zeros(self.dim, dtype=float)
        for c_i, b_i in zip(coords, self.vetores):
            resultado += c_i * b_i
        return resultado
    


if __name__ == "__main__":
    
    # Exemplo em R^2
    base_r2 = Base([[1, 1], 
                    [0, 2]])
    
    v = np.array([3, 2])

    coords = base_r2.coordenadas(v)
    v_reconstruido = base_r2.reconstruir(coords)

    print(f"Vetor original v:       {v}")
    print(f"Coordenadas [v]_B:      {coords}  ({coords[0]}*b1 + {coords[1]}*b2)")
    print(f"Vetor reconstruído:     {v_reconstruido}\n")
