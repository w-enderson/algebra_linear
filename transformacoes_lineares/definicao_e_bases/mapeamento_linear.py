import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
# ==============================
# transformações de R^n em R^m
# ==============================


class TransformacaoLinear:
    """
    Seja {b_1, ..., b_n} uma base de R^n,
    T: R^n -> R^m é completamente definido, em todo o domínio, a partir de {T(b_1), ..., T(b_n)}

    Já que:
    Para qualquer v = c_1*b_1 + ... + c_n*b_n em R^n:
        T(v) = c_1*T(b_1) + ... + c_n*T(b_n) em R^m

    (isso é garantido pelos axiomas de linearidade assumidos para T)
    """

    def __init__(
        self,
        base_dominio: Base,
        imagens_base: list[list[float] | np.ndarray],
    ):
        if len(imagens_base) != base_dominio.dim:
            raise ValueError(
                f"É necessário fornecer exatamente {base_dominio.dim} imagens, "
                "uma para cada vetor da base do domínio."
            )

        self.base = base_dominio
        self.imagens_base = [np.asarray(Tb, dtype=float) for Tb in imagens_base]

        # Infere a dimensão do contradomínio R^m a partir da primeira imagem
        self.dim_dominio = base_dominio.dim
        self.dim_contradominio = self.imagens_base[0].shape[0]

        for Tb in self.imagens_base:
            if Tb.ndim != 1 or Tb.shape[0] != self.dim_contradominio:
                raise ValueError(
                    f"Todas as imagens T(b_i) devem pertencer ao mesmo espaço R^{self.dim_contradominio}."
                )

    def __call__(self, v: list[float] | np.ndarray) -> np.ndarray:
        """
        Determina T(v) pela combinação linear sum(c_i*T(b_i)).
        """
        coords_rn = self.base.coordenadas(v)
        resultado = np.zeros(self.dim_contradominio, dtype=float)
        for c_i, Tb_i in zip(coords_rn, self.imagens_base):
            resultado += c_i * Tb_i
        return resultado


if __name__ == "__main__":
    # Exemplo: T: R^2 -> R^3
    base_r2 = Base([[1, 0], [1, 1]])

    T = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [2, 1, 0],   # T(1, 0)
            [0, 3, -1],  # T(1, 1)
        ],
    )

    v = np.array([4, 1])  # (4, 1) = 3*(1, 0) + 1*(1, 1)
    print(f"Coordenadas de v na base B: {base_r2.coordenadas(v)}")
    print(f"T({v}) = {T(v)}")  # 3*(2, 1, 0) + 1*(0, 3, -1) = (6, 6, -1)