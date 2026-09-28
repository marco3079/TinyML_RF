# TinyML Studio

Aplicativo PyQt5 para validar o dataset, treinar e quantizar a CNN, compilar o firmware e gravar o XIAO ESP32S3 Sense.

## Instalar

O aplicativo pode usar um Python diferente para o treino. Neste projeto, o ambiente TensorFlow 3.12 fica em `.training-env` e e selecionado automaticamente. Para recria-lo:

```powershell
conda create --prefix .training-env python=3.12 pip --yes
.training-env/python.exe -m pip install -r model/requirements.txt
```

Instale `desktop_app/requirements.txt` no Python usado para abrir a interface.

O PlatformIO Core deve estar instalado ou a extensao PlatformIO deve ter criado seu ambiente local.

## Executar

```powershell
python desktop_app/main.py
```

Selecione o Python que possui TensorFlow no campo **Python de treino**. A porta serial aparece depois que a placa e conectada. **Executar pipeline** realiza treino, exportacao INT8, compilacao e upload em sequencia.

## Monitor serial

A aba **Monitor serial** abre a porta selecionada e permite:

- escolher baud rate e conectar ou desconectar;
- visualizar dados com timestamps, auto-scroll ou formato hexadecimal;
- enviar texto, bytes HEX e escolher LF, CR+LF ou nenhum final de linha;
- solicitar o link da pagina com **Pedir IP**;
- reiniciar a placa, limpar a tela e salvar o log;
- acompanhar bytes recebidos e enviados.

Antes de gravar o firmware, o aplicativo libera automaticamente a porta serial e reconecta depois do upload.
