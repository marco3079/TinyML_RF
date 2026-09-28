# Modelo proprio de deteccao facial

> Estado: experimental. O firmware operacional usa ESP-DL de dois estagios porque a CNN minima treinada do zero ainda nao atingiu precisao suficiente em cenas reais.

Este projeto nao usa pesos faciais da Espressif nem um modelo pre-treinado. `train.py` cria uma CNN com pesos aleatorios, treina com o dataset fornecido e exporta os pesos para o firmware.

## Arquitetura

Entrada: imagem em tons de cinza com 64 x 64 pixels.

| Camada | Kernel | Passo | Saida |
| --- | --- | --- | --- |
| Conv2D + ReLU | 3 x 3, 4 filtros | 2 | 31 x 31 x 4 |
| Conv2D + ReLU | 3 x 3, 8 filtros | 2 | 15 x 15 x 8 |
| Conv2D + ReLU | 5 x 5, 12 filtros | 2 | 6 x 6 x 12 |
| Conv2D + sigmoid | 1 x 1, 5 filtros | 1 | 6 x 6 x 5 |

Cada celula produz `presenca, deslocamento_x, deslocamento_y, largura, altura`. Os kernels sao quantizados em INT8; vieses e ativacoes permanecem em ponto flutuante para manter o runtime pequeno e auditavel.

## Dataset

O treino aceita imagens JPG, JPEG ou PNG. Para cada imagem deve existir um TXT de mesmo nome no formato YOLO:

```text
0 centro_x centro_y largura altura
```

As coordenadas sao normalizadas entre 0 e 1 e a classe `0` significa face. Imagens sem face podem ter um TXT vazio ou nenhum TXT. Para um resultado minimamente util, recomenda-se milhares de imagens com variacao de iluminacao, distancia, pose e tom de pele.

### WIDER FACE

Baixe as imagens e anotacoes do WIDER FACE. Para o conjunto de treino:

```powershell
python model/prepare_wider.py `
  --images C:/datasets/WIDER_train/images `
  --annotations C:/datasets/wider_face_split/wider_face_train_bbx_gt.txt
```

O conversor ignora caixas marcadas como invalidas e rostos menores que 8 pixels.

## Treinar e exportar

Use Python 3.11 ou 3.12, pois o TensorFlow pode nao oferecer pacote para versoes mais novas:

```powershell
python -m venv .venv-training
.venv-training/Scripts/Activate.ps1
python -m pip install -r model/requirements.txt
python model/train.py --dataset C:/datasets/WIDER_train/images --validate-only
python model/train.py --dataset C:/datasets/WIDER_train/images --epochs 60
```

O treino gera `src/model_data.h`, com os pesos INT8 compilados no firmware, e `src/model_data.keras`, checkpoint para analises no computador. Depois do treino, recompile e grave o firmware; nao e necessario copiar um modelo separadamente para a placa.
