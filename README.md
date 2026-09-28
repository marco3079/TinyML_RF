# Deteccao facial no XIAO ESP32S3 Sense

Detector facial local com camera OV2640, Wi-Fi e pagina web servida pelo proprio XIAO. O firmware operacional usa o detector ESP-DL de dois estagios, com suporte a multiplas faces e supressao de caixas duplicadas. O frame RGB565 recebe as caixas amarelas e e convertido para JPEG antes de ser enviado ao navegador como MJPEG.

O diretorio `model` preserva um laboratorio de treinamento do zero para fins didaticos. As avaliacoes em cenas reais mostraram que a CNN minima experimental ainda nao tem precisao suficiente para controlar o detector embarcado.

## Aplicativo desktop

Execute `desktop_app/start.bat` para abrir o TinyML Studio. A interface permite selecionar e validar o dataset, configurar epocas e lote, treinar, quantizar o modelo, compilar o firmware, escolher a porta serial e gravar a placa. O log e o progresso de cada processo aparecem na propria janela.

## Hardware

- Seeed Studio XIAO ESP32S3 Sense;
- camera OV2640 do kit Sense;
- antena Wi-Fi conectada ao pequeno conector IPEX;
- cabo USB de dados.

## Preparar e compilar

1. Instale a extensao **PlatformIO IDE** no VS Code.
2. Duplique `include/secrets.h.example` como `include/secrets.h`.
3. Preencha `WIFI_SSID` e `WIFI_PASSWORD`. O XIAO aceita apenas Wi-Fi de 2,4 GHz.
4. Conecte a placa e execute **PlatformIO: Upload** na barra inferior do VS Code.
5. Abra a aba **Monitor serial** do TinyML Studio, conecte em 115200 baud e clique em **Pedir IP**.
6. Se a rede configurada conectar, abra o link informado, por exemplo `http://192.168.1.50`, em um dispositivo na mesma rede.
7. Se a conexao falhar em 15 segundos, conecte o computador ou celular a rede `TinyML-Camera` e abra `http://192.168.4.1`.

Pela linha de comando, depois de instalar o PlatformIO Core:

```powershell
pio run
pio run --target upload
pio device monitor --baud 115200
```

Se o upload nao iniciar, segure **BOOT**, toque em **RESET**, solte **BOOT** e tente novamente. O ambiente ja habilita a PSRAM OPI necessaria para camera e modelo.

## Onde ficam firmware e modelo

O firmware esta em `src/main.cpp`, a integracao ESP-DL em `src/face_detector.cpp` e a pagina em `src/web_page.h`. Os pesos ESP-DL sao ligados ao `firmware.bin` pelo pacote Arduino-ESP32; codigo, pagina e detector seguem juntos no Upload.

## Limites esperados

A inferencia e o JPEG disputam CPU, entao o stream prioriza deteccao e nao video fluido. Iluminacao frontal, distancia de 0,5 a 2 m e rosto com pelo menos cerca de 50 pixels melhoram o resultado. Este projeto faz deteccao, nao reconhecimento de identidade.
