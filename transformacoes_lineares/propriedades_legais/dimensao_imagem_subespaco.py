from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# =================================================
# Exercício 4.18 do Elon Lages Lima (Álgebra Linear)
# =================================================
# Se E' <= E é um subespaço de dimensão finita e A: E -> F é linear, então
# A(E') <= F tem dimensão finita e dim A(E') <= dim E'.
#
# Ideia da demonstração:
#   Se {u_1, ..., u_k} é uma base de E' (dim E' = k), então todo w em A(E')
#   se escreve como w = A(sum c_i u_i) = sum c_i A(u_i). Logo, {A(u_1), ..., A(u_k)}
#   gera A(E'), o que garante dim A(E') <= k = dim E', já que não necessariamente são LI.
#
# Experimento com E' = R^2 (dim = 2) e A: R^2 -> R dada por A(x, y) = x - y:
#   A(1, 0) = (1)
#   A(0, 1) = (-1)
#   Vários vetores espalhados pelo plano R^2 (dim 2) são mapeados na reta R (dim 1).

if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r2 = Base([[1.0, 0.0], [0.0, 1.0]])

    # Funcional linear A: R^2 -> R definido por A(x, y) = x - y
    A = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [1.0],   # A(1, 0) = 1
            [-1.0],  # A(0, 1) = -1
        ],
    )

    # Plota a estrutura base do mapeamento R^2 -> R
    fig, (ax_dom, ax_cod) = plotar_mapeamento(
        A,
        alcance_grade=3,
        titulo=r"Exercício 4.18 (Elon): $E' = \mathbb{R}^2$ ($\dim = 2$) mapeado em $A(E') \subset \mathbb{R}$ ($\dim = 1$)",
    )

    # Gera uma grade densa de vários vetores espalhados por todo o plano E' = R^2
    coords = np.linspace(-2.5, 2.5, 9)
    gx, gy = np.meshgrid(coords, coords)
    vetores_r2 = np.column_stack([gx.ravel(), gy.ravel()])

    # Calcula A(v) para cada vetor v de R^2
    imagens_r = np.array([A(v)[0] for v in vetores_r2])

    # Colore cada ponto pelo valor de sua imagem A(v) = x - y (evidencia as curvas de nível)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=imagens_r.min(), vmax=imagens_r.max())

    ax_dom.scatter(
        vetores_r2[:, 0],
        vetores_r2[:, 1],
        c=imagens_r,
        cmap=cmap,
        norm=norm,
        s=45,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=rf"{len(vetores_r2)} vetores em $E' = \mathbb{{R}}^2$ ($\dim = 2$)",
    )

    # Plota todos os pontos mapeados sobre o subespaço 1D no contradomínio
    ax_cod.axhline(0, color="#6366f1", lw=4.0, alpha=0.25, zorder=2, label=r"$A(E') = \mathrm{span}\{1, -1\} = \mathbb{R}$ ($\dim = 1$)")
    ax_cod.scatter(
        imagens_r,
        np.zeros_like(imagens_r),
        c=imagens_r,
        cmap=cmap,
        norm=norm,
        s=65,
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=r"Imagens $A(v)$ colapsadas em 1D",
    )

    # Ajusta o limite horizontal do contradomínio para caber todos os valores de x - y (até +-5)
    ax_cod.set_xlim(-5.5, 5.5)

    ax_dom.legend(loc="upper left", fontsize=8.5)
    ax_cod.legend(loc="upper left", fontsize=8.5)

 
    fig.savefig(pasta_graficos / "dimensao_imagem_subespaco.png", dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em: {pasta_graficos}")
    plt.show()