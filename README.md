## TCC Etec 2025 - Reconhecimento Facial
Projeto de Reconhecimento Facial e Integração

## Grupo
- Giovanni
- Gustavo
- João
- Guilherme
- Danilo

## TODO
- Armazenar fotos de rostos conhecidos
- Deve ser criado script pra capturar jpgs de rostos e deixar em um diretório

- Abrir software de captura de video 
-- Capturar frames a cada 1s e verificar se neste frame há uma face
--- Se houver uma face, chamar a biblioteca de face_recognization e verificar se é um rosto conhecido

## TODO Pesquisa Teórica do TCC
- Falar o que é a OpenCV, no caso, utilizamos para capturar video e salvar frames
- Falar sobre a importância de detecção de faces a partir de um modelo e falar o que faz a biblioteca YOLO
- Entender o que faz a face_recognaztion e falar sobre as técnicas que ela utiliza internamente
- Por exemplo vi que na instalação usa pyTorch, acredito que ela usa rede neural, por exemplo

## Criar ambiente Virtual
Recomenda-se criar ambiente virutal em quanto estiver desenvolvendo e, daí, a pasta .venv pode ser excluída
- python -m venv .venv, para criar
- E, para ativar, antes de desenvolver, usar o .venv\Scripts\activate.bat

## Instalar o CMake
- Instalar o Visual Studio for C++
- Instalar o https://cmake.org/download/
-- Selecionar Windows x64 Installer:

## Para instalar bibliotecas
Executar o comando pip install -r .\requeriments.txt 


## Leituras
-  https://medium.com/@erdibillkaan/object-detection-and-face-recognition-with-yolov8n-356b22bacc48
-  https://www.perplexity.ai/search/no-python-estou-instalando-a-b-jygrmL.wQo2uBVRMPe7S0g

## Sobre a parte teórica

- Deve ser falado o que é Visão Computacional, fase que vão desde a aquisição da imagem até a extração de informação da mesama
- Falar sobre as bibliotecas do OpenCV, biblioteca Yolo para extração de objetos (no caso Faces), biblioteca face_detection para detecção de rosto
- explicar como funciona por cima a face_dectection, para buscar
- Verificar se as mesmas usam a matriz de distância euclidiana, pra fazer o reconhecimento
- Tentar fazer um teste e anotar se o uso da Memória e CPU de uma máquina reconhecimento 5 pessoas, é o mesmo que 50... Ou seja,
avaliar as limitações técnicas, o professor pode perguntar.
- Vocês tem que estudar matriz confusão (positivo-real, etc)... https://developers.google.com/machine-learning/crash-course/classification/thresholding?hl=pt-br
- Tem que demonstrar se o projeto de vocês é confiável ou não, baseado no reconhecimento real e verdadeiro das pessoas, e, com isso, só fazendo testes e anotações.. Pedir para os outros colegas da classe participar e, vocês anotam
- Abaixo 2 TCCs que explicam a parte de Visão Computacional, conceitos como subtração de fundo, etc
- Lembrando que eu acho que o face_detection e o Yolo são coisas diferentes mas, que são bibliotecas que ajudam no projeto de Visão Computacional. Uma coisa é detectar um objeto face, daí utilizamos o Yolo. E, outra coisa, é fazer a detecção desta face em uma base de cadastro nossa (as imagens cadastradas).
- A partir do momento que temos a informação, ou seja, que uma pessoa foi encontrada, daí podemos gravar Log, enviar SMS, abrir catraca, daí utilizando outras bibliotecas.

Material de exemplo pra estudar:
- https://github.com/marciofcruz/Python-Estudos/blob/master/ReconhecimentoSinaisLibras/doc/TCC%20Reconhecimento%20Sinais%20de%20Libras%20-%20Vers%C3%A3o%201.pdf
- https://github.com/marciofcruz/Python-Estudos/blob/master/ReconhecimentoSinaisLibras/doc/TCC%20Reconhecimento%20Sinais%20de%20Libras%20-%20Vers%C3%A3o%202.pdf