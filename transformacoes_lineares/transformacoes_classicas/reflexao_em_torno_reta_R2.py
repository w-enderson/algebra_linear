from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# ===============================================
# Exemplo 4.4 do Elon Lages Lima (Álgebra Linear)
# ===============================================
# Reflexão ortogonal R: R^2 -> R^2 em torno da reta y = a*x:
#   x' = ((1 - a^2) / (1 + a^2)) * x + (2*a / (1 + a^2)) * y
#   y' = (2*a / (1 + a^2)) * x + ((a^2 - 1) / (1 + a^2)) * y
#
# Aplicando diretamente nos vetores da base canônica e_1 = (1, 0) e e_2 = (0, 1):
#   R(1, 0) = ((1 - a^2) / (1 + a^2), 2*a / (1 + a^2))
#   R(0, 1) = (2*a / (1 + a^2), (a^2 - 1) / (1 + a^2))

# obtidos a partir do fato de que R = 2*P - I, 
#   P= projeção ortogonal sobre a reta y = a*x e 
#   I= transformação identidade.

def desenhar_reta(ax_dom, ax_cod, a: float, limite: float = 4.0):
    """Desenha a reta y = a*x (eixo de simetria/reflexão) no domínio e no contradomínio."""
    x = np.linspace(-limite, limite, 100)
    y = a * x
    for ax in (ax_dom, ax_cod):
        ax.plot(x, y, linestyle="--", color="#f59e0b", lw=1.8, label=rf"Espelho $y = {a:g}x$", zorder=3)
        ax.legend(loc="upper left", fontsize=8.5)


if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r2 = Base([[1.0, 0.0], [0.0, 1.0]])

    # Caso 1: Reflexão em torno do eixo x (reta y = 0 -> a = 0)
    a1 = 0.0
    R_eixo_x = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [(1 - a1**2) / (1 + a1**2), (2 * a1) / (1 + a1**2)],
            [(2 * a1) / (1 + a1**2), (a1**2 - 1) / (1 + a1**2)],
        ],
    )

    v1 = np.array([2.0, 1.5])
    fig1, (ax_dom1, ax_cod1) = plotar_mapeamento(
        R_eixo_x,
        v=v1,
        alcance_grade=3,
        titulo=r"Exemplo 4.4 (Elon): Reflexão em torno de $y = 0$ ($a = 0$)",
    )
    desenhar_reta(ax_dom1, ax_cod1, a1)
    fig1.savefig(pasta_graficos / "reflexao_reta_eixo_x.png", dpi=300, bbox_inches="tight")

    # Caso 2: Reflexão em torno da reta y = x (a = 1)
    a2 = 1.0
    R_diagonal = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [(1 - a2**2) / (1 + a2**2), (2 * a2) / (1 + a2**2)],
            [(2 * a2) / (1 + a2**2), (a2**2 - 1) / (1 + a2**2)],
        ],
    )

    v2 = np.array([2.5, 1.0])
    fig2, (ax_dom2, ax_cod2) = plotar_mapeamento(
        R_diagonal,
        v=v2,
        alcance_grade=3,
        titulo=r"Exemplo 4.4 (Elon): Reflexão em torno de $y = x$ ($a = 1$)",
    )
    desenhar_reta(ax_dom2, ax_cod2, a2)
    fig2.savefig(pasta_graficos / "reflexao_reta_diagonal.png", dpi=300, bbox_inches="tight")

    plt.show()