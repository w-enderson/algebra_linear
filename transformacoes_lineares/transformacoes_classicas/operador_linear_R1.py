from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# ===============================================
# Exemplo 4.1 do Elon Lages Lima (Álgebra Linear)
# ===============================================
# Todo operador linear T: R -> R é completamente determinado pela
# imagem do único vetor da base: T(x) = a * x.

if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r1 = Base([[1.0]])

    # Caso 1: Dilatação com inversão de orientação (T(1) = -2)
    T_inversao = TransformacaoLinear(
        base_dominio=base_r1,
        imagens_base=[[-2.0]],
    )

    x = np.array([1.5])
    print(f"T(1) = {T_inversao([1.0])[0]:g}")
    print(f"T({x[0]:g}) = {T_inversao(x)[0]:g}")

    fig1, _ = plotar_mapeamento(
        T_inversao,
        v=x,
        alcance_grade=3,
        titulo=r"Exemplo 4.1 (Elon): Operador Linear $T: \mathbb{R} \to \mathbb{R}$ ($a = -2$)",
    )
    fig1.savefig(pasta_graficos / "operador_linear_r1_inversao.png", dpi=300, bbox_inches="tight")

    # Caso 2: Contração positiva (T(1) = 0.5)
    T_contracao = TransformacaoLinear(
        base_dominio=base_r1,
        imagens_base=[[0.5]],
    )

    fig2, _ = plotar_mapeamento(
        T_contracao,
        v=[3.0],
        alcance_grade=3,
        titulo=r"Exemplo 4.1 (Elon): Operador Linear $T: \mathbb{R} \to \mathbb{R}$ ($a = 0.5$)",
    )
    fig2.savefig(pasta_graficos / "operador_linear_r1_contracao.png", dpi=300, bbox_inches="tight")

    print(f"Gráficos salvos em: {pasta_graficos}")
    plt.show()