from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformacoes_lineares.definicao_e_bases.base_e_coordenadas import Base
from transformacoes_lineares.definicao_e_bases.mapeamento_linear import TransformacaoLinear
from transformacoes_lineares.definicao_e_bases.viz_base import plotar_mapeamento

# ===============================================
# Exemplo 4.2 do Elon Lages Lima (Álgebra Linear)
# ===============================================
# Rotação de um ângulo theta (anti-horário) em torno da origem no R^2:
#   T(e_1) = (cos(theta), sen(theta))
#   T(e_2) = (-sen(theta), cos(theta))

# T(e_1) e T(e_2) são obtidos via trigonometria;

if __name__ == "__main__":
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    base_r2 = Base([[1.0, 0.0], [0.0, 1.0]])

    # Caso 1: Rotação de 45° (pi/4 radianos) no sentido anti-horário
    theta1 = np.pi / 4
    T_rot_45 = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [np.cos(theta1), np.sin(theta1)],
            [-np.sin(theta1), np.cos(theta1)],
        ],
    )

    v = np.array([2.0, 1.0])
    print(f"Rotação 45° -> T({v}) = {np.round(T_rot_45(v), 4)}")

    fig1, _ = plotar_mapeamento(
        T_rot_45,
        v=v,
        alcance_grade=3,
        titulo=r"Exemplo 4.2 (Elon): Rotação em torno da origem ($\theta = 45^\circ$)",
    )
    fig1.savefig(pasta_graficos / "rotacao_origem_r2_45graus.png", dpi=300, bbox_inches="tight")

    # Caso 2: Rotação de 90° (pi/2 radianos) -> T(e1) = (0, 1) e T(e2) = (-1, 0)
    theta2 = np.pi / 2
    T_rot_90 = TransformacaoLinear(
        base_dominio=base_r2,
        imagens_base=[
            [np.cos(theta2), np.sin(theta2)],
            [-np.sin(theta2), np.cos(theta2)],
        ],
    )

    print(f"Rotação 90° -> T({v}) = {np.round(T_rot_90(v), 4)}")

    fig2, _ = plotar_mapeamento(
        T_rot_90,
        v=v,
        alcance_grade=3,
        titulo=r"Exemplo 4.2 (Elon): Rotação em torno da origem ($\theta = 90^\circ$)",
    )
    fig2.savefig(pasta_graficos / "rotacao_origem_r2_90graus.png", dpi=300, bbox_inches="tight")

    print(f"Gráficos salvos em: {pasta_graficos}")
    plt.show()