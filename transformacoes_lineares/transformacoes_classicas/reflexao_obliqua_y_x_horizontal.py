from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# =================================================
# Exercício 4.12 do Elon (Álgebra Linear)
# =================================================
# Operador A: R^2 -> R^2 tal que o segmento ligando v = (x, y) a A(v) = (x', y')
# é horizontal (y' = y) e seu ponto médio ((x + x')/2, y) está sobre a reta y = x:
#   (x + x') / 2 = y  =>  x' = -x + 2y
#   y' = y
#
# Aplicando nos vetores da base canônica:
#   A(1, 0) = (-1, 0)
#   A(0, 1) = (2, 1)


def desenhar_reflexao_horizontal(ax_dom, ax_cod, v: np.ndarray, limite: float = 4.0):
    """Desenha a reta y = x, o segmento horizontal de v até A(v) e o ponto médio."""
    x = np.linspace(-limite, limite, 100)

    # Reta y = x nos dois painéis (lugar geométrico dos pontos fixos e pontos médios)
    for ax in (ax_dom, ax_cod):
        ax.plot(x, x, linestyle="--", color="#f59e0b", lw=1.8, label=r"Reta $y = x$ (ponto médio)", zorder=3)

    # Segmento horizontal ligando v = (x, y) até A(v) = (-x + 2y, y) no domínio
    av = np.array([-v[0] + 2.0 * v[1], v[1]])
    pm = 0.5 * (v + av)  # Ponto médio (y, y) sobre a reta y = x

    ax_dom.plot([v[0], av[0]], [v[1], av[1]], linestyle=":", color="#8b5cf6", lw=1.6, zorder=3)
    ax_dom.scatter([av[0]], [av[1]], color="#8b5cf6", s=35, zorder=5)
    ax_dom.scatter([pm[0]], [pm[1]], color="#f59e0b", edgecolor="black", s=45, zorder=6, label="Ponto médio")

    ax_dom.legend(loc="upper left", fontsize=8.5)
    ax_cod.legend(loc="upper left", fontsize=8.5)


if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r2 = Base([[1.0, 0.0], [0.0, 1.0]])

    A_reflexao_horiz = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [-1.0, 0.0],  # A(1, 0)
            [2.0, 1.0],   # A(0, 1)
        ],
    )

    # Caso 1: v = (2.5, 1.0) -> A(v) = (-0.5, 1.0), ponto médio = (1.0, 1.0)
    v1 = np.array([2.5, 1.0])
    print(f"A({v1}) = {np.round(A_reflexao_horiz(v1), 4)}")
    print(f"Involução A(A(v1)) = {np.round(A_reflexao_horiz(A_reflexao_horiz(v1)), 4)}")

    fig1, (ax_dom1, ax_cod1) = plotar_mapeamento(
        A_reflexao_horiz,
        v=v1,
        alcance_grade=3,
        titulo=r"Exercício 4.12 (Elon): Segmento horizontal com ponto médio em $y = x$",
    )
    desenhar_reflexao_horizontal(ax_dom1, ax_cod1, v1)
    fig1.savefig(pasta_graficos / "reflexao_horizontal_y_x_caso1.png", dpi=300, bbox_inches="tight")

    # Caso 2: v = (-1.0, 1.5) -> A(v) = (4.0, 1.5), ponto médio = (1.5, 1.5)
    v2 = np.array([-1.0, 1.5])
    print(f"A({v2}) = {np.round(A_reflexao_horiz(v2), 4)}")

    fig2, (ax_dom2, ax_cod2) = plotar_mapeamento(
        A_reflexao_horiz,
        v=v2,
        alcance_grade=4,
        titulo=r"Exercício 4.12 (Elon): Segmento horizontal com ponto médio em $y = x$",
    )
    desenhar_reflexao_horizontal(ax_dom2, ax_cod2, v2, limite=5.0)
    fig2.savefig(pasta_graficos / "reflexao_horizontal_y_x_caso2.png", dpi=300, bbox_inches="tight")

    print(f"Gráficos salvos em: {pasta_graficos}")
    plt.show()