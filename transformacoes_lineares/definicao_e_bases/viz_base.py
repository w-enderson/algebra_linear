import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear


def _limpar_num(val: float, casas: int = 2) -> float:
    """Arredonda e zera ruídos numéricos de ponto flutuante (ex: 6.12e-17 -> 0)."""
    arredondado = round(float(val), casas)
    return 0.0 if abs(arredondado) < 10 ** (-casas) else arredondado


def _formatar_vetor(vec: np.ndarray, casas: int = 2) -> str:
    """Formata um vetor como '(x, y, ...)' arredondado e sem '.0' desnecessário."""
    coords = ", ".join(f"{_limpar_num(val, casas):g}" for val in vec)
    return f"({coords})"


def _configurar_eixo(ax, dim: int, limite: int, titulo: str):
    ax.set_xlim(-limite, limite)
    ax.set_title(titulo, fontsize=11, pad=10)

    if dim == 1:
        ax.axhline(0, color="black", lw=1.0, zorder=2)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    elif dim == 2:
        ax.set_ylim(-limite, limite)
        ax.axhline(0, color="black", lw=0.8, zorder=2)
        ax.axvline(0, color="black", lw=0.8, zorder=2)
        ax.set_aspect("equal")
    elif dim == 3:
        ax.set_ylim(-limite, limite)
        ax.set_zlim(-limite, limite)
        ax.set_box_aspect([1, 1, 1])


def _desenhar_vetor(ax, vetor: np.ndarray, cor: str, label: str, lw: float = 1.8):
    dim = len(vetor)
    if dim == 1:
        ax.quiver(
            0, 0, vetor[0], 0,
            angles="xy", scale_units="xy", scale=1,
            color=cor, width=0.008, label=label, zorder=4
        )
        ax.scatter([vetor[0]], [0], color=cor, s=35, zorder=5)
    elif dim == 2:
        ax.quiver(
            0, 0, vetor[0], vetor[1],
            angles="xy", scale_units="xy", scale=1,
            color=cor, width=0.008, label=label, zorder=4
        )
    elif dim == 3:
        ax.quiver(
            0, 0, 0, vetor[0], vetor[1], vetor[2],
            color=cor, lw=lw, arrow_length_ratio=0.12, label=label
        )


def _desenhar_malha(ax, vetores_geradores: list[np.ndarray], passos: np.ndarray, t: np.ndarray, cor: str):
    """Desenha as combinações lineares c*u_i + t*u_j em 1D, 2D ou 3D."""
    dim_espaco = len(vetores_geradores[0])
    n_geradores = len(vetores_geradores)

    if dim_espaco == 1:
        for u in vetores_geradores:
            pts = passos * u[0]
            ax.scatter(pts, np.zeros_like(pts), color=cor, marker="|", s=80, zorder=2)
        return

    if n_geradores == 1:
        pts = t[None, :] * vetores_geradores[0][:, None]
        ax.plot(*pts, color=cor, lw=1.0, alpha=0.8)
        return

    for i in range(n_geradores):
        for c in passos:
            for j in range(n_geradores):
                if i == j:
                    continue
                pts = c * vetores_geradores[i][:, None] + t[None, :] * vetores_geradores[j][:, None]
                ax.plot(*pts, color=cor, lw=0.7, alpha=0.75)


def plotar_mapeamento(
    T: TransformacaoLinear,
    v: list[float] | np.ndarray | None = None,
    alcance_grade: int = 3,
    casas_decimais: int = 2,
    titulo: str = "Transformação Linear T: V -> W",
):
    """Visualiza mapeamentos lineares onde dim(V) e dim(W) estão em {1, 2, 3}."""
    n, m = T.dim_dominio, T.dim_contradominio
    if n not in (1, 2, 3) or m not in (1, 2, 3):
        raise NotImplementedError(
            f"Visualização gráfica suportada apenas para dimensões 1, 2 e 3 "
            f"(recebido: R^{n} -> R^{m})."
        )

    fig = plt.figure(figsize=(13.5, 6.2))
    ax_dom = fig.add_subplot(1, 2, 1, projection="3d" if n == 3 else None)
    ax_cod = fig.add_subplot(1, 2, 2, projection="3d" if m == 3 else None)

    cores_base = ["#ef4444", "#10b981", "#f59e0b"]
    passos = np.arange(-alcance_grade, alcance_grade + 1)
    t = np.linspace(-alcance_grade, alcance_grade, 60)

    # 1. Desenha a malha gerada pelas combinações lineares dos vetores da base
    _desenhar_malha(ax_dom, T.base.vetores, passos, t, cor="#d1d5db")
    _desenhar_malha(ax_cod, T.imagens_base, passos, t, cor="#93c5fd")

    # 2. Desenha cada vetor da base b_i e sua respectiva imagem T(b_i) com seus valores
    for i, (b_i, Tb_i) in enumerate(zip(T.base.vetores, T.imagens_base)):
        cor = cores_base[i % len(cores_base)]
        _desenhar_vetor(ax_dom, b_i, cor, rf"$b_{i+1} = {_formatar_vetor(b_i, casas_decimais)}$")
        _desenhar_vetor(ax_cod, Tb_i, cor, rf"$T(b_{i+1}) = {_formatar_vetor(Tb_i, casas_decimais)}$")

    # 3. Se fornecido um vetor v, plota v e T(v)
    if v is not None:
        v = np.asarray(v, dtype=float)
        coords = T.base.coordenadas(v)
        Tv = T(v)

        lbl_v = " + ".join(f"{_limpar_num(c, casas_decimais):g}b_{i+1}" for i, c in enumerate(coords))
        lbl_Tv = " + ".join(f"{_limpar_num(c, casas_decimais):g}T(b_{i+1})" for i, c in enumerate(coords))

        _desenhar_vetor(
            ax_dom, v, "#8b5cf6",
            rf"$v = {_formatar_vetor(v, casas_decimais)} = {lbl_v}$", lw=2.2
        )
        _desenhar_vetor(
            ax_cod, Tv, "#8b5cf6",
            rf"$T(v) = {_formatar_vetor(Tv, casas_decimais)} = {lbl_Tv}$", lw=2.2
        )

    # Monta subtítulos explícitos com os conjuntos da base e das imagens
    str_base = ", ".join(_formatar_vetor(b, casas_decimais) for b in T.base.vetores)
    str_imagens = ", ".join(_formatar_vetor(Tb, casas_decimais) for Tb in T.imagens_base)

    limite = alcance_grade + 1
    _configurar_eixo(
        ax_dom, n, limite,
        rf"Domínio ($\mathbb{{R}}^{n}$)" + f"\nBase $\\mathcal{{B}} = \\{{{str_base}\\}}$"
    )
    _configurar_eixo(
        ax_cod, m, limite,
        rf"Contradomínio ($\mathbb{{R}}^{m}$)" + f"\nImagens $T(\\mathcal{{B}}) = \\{{{str_imagens}\\}}$"
    )

    ax_dom.legend(loc="upper left", fontsize=8.5)
    ax_cod.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, (ax_dom, ax_cod)