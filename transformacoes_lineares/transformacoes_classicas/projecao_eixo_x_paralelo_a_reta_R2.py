from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# =================================================
# Exercício 4.4 do Elon Lages Lima (Álgebra Linear)
# =================================================
# Projeção P: R^2 -> R^2 sobre o eixo x, paralelamente à reta y = a*x (a != 0):
#   x' = x - (1 / a) * y
#   y' = 0
#
# Aplicando diretamente nos vetores da base canônica:
#   P(1, 0) = (1, 0)
#   P(0, 1) = (-1 / a, 0)


def desenhar_direcao_projecao(ax_dom, ax_cod, a: float, v: np.ndarray, limite: float = 4.0):
    """Desenha a reta diretora y = a*x (núcleo) e o deslocamento paralelo de v até P(v)."""
    x = np.linspace(-limite, limite, 100)
    y = a * x

    # Reta diretora y = a*x no domínio (todos os seus pontos vão para (0, 0))
    ax_dom.plot(x, y, linestyle="--", color="#f59e0b", lw=1.8, label=rf"Direção $y = {a:g}x$ ($\ker P$)", zorder=3)

    # Segmento paralelo à reta y = a*x ligando v até sua projeção P(v) no eixo x
    pv = np.array([v[0] - v[1] / a, 0.0])
    ax_dom.plot([v[0], pv[0]], [v[1], pv[1]], linestyle=":", color="#8b5cf6", lw=1.6, zorder=3)
    ax_dom.scatter([pv[0]], [pv[1]], color="#8b5cf6", s=35, zorder=5)

    ax_dom.legend(loc="upper left", fontsize=8.5)
    ax_cod.legend(loc="upper left", fontsize=8.5)


if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r2 = Base([[1.0, 0.0], [0.0, 1.0]])

    # Caso 1: Projeção sobre o eixo x paralelamente à reta y = x (a = 1)
    # P(1, 0) = (1, 0) e P(0, 1) = (-1, 0)
    a1 = 1.0
    P_par_diagonal = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [1.0, 0.0],        # P(1, 0)
            [-1.0 / a1, 0.0],  # P(0, 1)
        ],
    )

    v1 = np.array([2.5, 1.5])
    print(f"Projeção paralela a y = x (a = 1) -> P({v1}) = {np.round(P_par_diagonal(v1), 4)}")

    fig1, (ax_dom1, ax_cod1) = plotar_mapeamento(
        P_par_diagonal,
        v=v1,
        alcance_grade=3,
        titulo=r"Exercício 4.4 (Elon): Projeção sobre o eixo $x$ paralela a $y = x$ ($a = 1$)",
    )
    desenhar_direcao_projecao(ax_dom1, ax_cod1, a1, v1)
    fig1.savefig(pasta_graficos / "projecao_eixo_x_paralela_y_x.png", dpi=300, bbox_inches="tight")

    # Caso 2: Projeção sobre o eixo x paralelamente à reta y = 2x (a = 2)
    # P(1, 0) = (1, 0) e P(0, 1) = (-0.5, 0)
    a2 = 2.0
    P_par_2x = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [1.0, 0.0],        # P(1, 0)
            [-1.0 / a2, 0.0],  # P(0, 1)
        ],
    )

    v2 = np.array([1.0, 2.0])
    print(f"Projeção paralela a y = 2x (a = 2) -> P({v2}) = {np.round(P_par_2x(v2), 4)}")

    fig2, (ax_dom2, ax_cod2) = plotar_mapeamento(
        P_par_2x,
        v=v2,
        alcance_grade=3,
        titulo=r"Exercício 4.4 (Elon): Projeção sobre o eixo $x$ paralela a $y = 2x$ ($a = 2$)",
    )
    desenhar_direcao_projecao(ax_dom2, ax_cod2, a2, v2)
    fig2.savefig(pasta_graficos / "projecao_eixo_x_paralela_y_2x.png", dpi=300, bbox_inches="tight")

    # Caso 3: Reta abaixo do vetor v -> y = 0.5x (a = 0.5) e v = (1.5, 2.5)
    # Como o vetor está acima da reta diretora, a projeção cruza para o semieixo negativo:
    # P(1, 0) = (1, 0) e P(0, 1) = (-2, 0) => P(1.5, 2.5) = (-3.5, 0)
    a3 = 0.5
    P_par_reta_abaixo = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [1.0, 0.0],        # P(1, 0)
            [-1.0 / a3, 0.0],  # P(0, 1) = (-2, 0)
        ],
    )

    v3 = np.array([1.5, 2.5])
    print(f"Projeção paralela a y = 0.5x (a = 0.5) -> P({v3}) = {np.round(P_par_reta_abaixo(v3), 4)}")
    print(f"Idempotência P(P(v)) = {np.round(P_par_reta_abaixo(P_par_reta_abaixo(v3)), 4)}")

    fig3, (ax_dom3, ax_cod3) = plotar_mapeamento(
        P_par_reta_abaixo,
        v=v3,
        alcance_grade=4,
        titulo=r"Exercício 4.4 (Elon): Projeção sobre o eixo $x$ paralela a $y = 0.5x$ ($a = 0.5$)",
    )
    desenhar_direcao_projecao(ax_dom3, ax_cod3, a3, v3, limite=5.0)
    fig3.savefig(pasta_graficos / "projecao_eixo_x_paralela_y_05x.png", dpi=300, bbox_inches="tight")

    print(f"Gráficos salvos em: {pasta_graficos}")
    plt.show()