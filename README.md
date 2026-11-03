<div align="center">

# ⚽ FC-Tactical-Engine

### *Real-Time Spatial Telemetry & Tactical AI Assistant for EA Sports FC*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Windows 10/11](https://img.shields.io/badge/Windows-10%20%2F%2011%20(64--bit)-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![DirectX DDA](https://img.shields.io/badge/DirectX-Desktop%20Duplication%20API-107C10?style=for-the-badge&logo=xbox&logoColor=white)](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api)
[![DuckDB](https://img.shields.io/badge/DuckDB-Columnar%20OLAP-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status: Em Desenvolvimento Ativo](https://img.shields.io/badge/Status-Em%20Desenvolvimento%20Ativo-22C55E?style=for-the-badge)]()

<p align="center">
  <b>Motor de visão computacional de ultrabaixa latência (< 15ms end-to-end), geometria computacional espacial (Diagramas de Voronoi) e telemetria analítica colunar para tomada de decisão em tempo real no EA Sports FC.</b>
</p>

[Visão Funcional](#-visão-funcional-o-que-a-aplicação-faz) •
[Arquitetura & Segurança](#️-arquitetura-técnica--segurança) •
[Pré-requisitos](#-pré-requisitos-de-sistema) •
[Instalação](#-guia-de-instalação-e-execução-powershell) •
[Estrutura](#-estrutura-do-repositório-src-layout) •
[Roadmap](#-roadmap--status-do-projeto)

</div>

---

## 🎯 Visão Funcional (O que a aplicação faz)

Durante uma partida competitiva de **EA Sports FC**, o jogador humano sofre de um gargalo cognitivo natural: enquanto foca a visão periférica na disputa de bola imediata (microjogo), vulnerabilidades críticas ou oportunidades de ruptura acontecem fora do seu foco de atenção primário (macrojogo).

O **FC-Tactical-Engine** atua como um **Auxiliar Técnico Virtual em Tempo Real** (*"Spotter / Coach Virtual"*), inspecionando continuamente os elementos visuais da interface (HUD e Radar 2D) a **60 FPS** para converter pixels brutos em inteligência tática acionável:

* 📡 **Leitura Contínua do Minimap/Radar e HUD:** Rastreamento em tempo real dos centróides `(X, Y)` dos 22 jogadores e da bola no plano cartesiano normalizado `[0.0, 1.0]²`, aliado à detecção automática do estado da partida (`EM_ANDAMENTO`, `BOLA_PARADA`, `REPLAY`, `PAUSA`) e leitura de placar/cronômetro.
* 📐 **Análise de Dominância Espacial (Diagramas de Voronoi):** Particionamento geométrico contínuo do gramado para quantificar o controle territorial de cada equipe, medir a compactação de linhas (distância entre zaga e ataque) e detectar buracos críticos entre os zagueiros centrais (`CENTRO_EXPOSTO`).
* 🎯 **Detecção de Linhas de Passe Abertas:** Cálculo vetorial a partir do portador da bola para todos os receptores elegíveis, avaliando cones de sombra e polígonos de interceptação dos marcadores adversários para identificar passes de ruptura livres de marcação.
* 🎧 **Avisos Contextuais por Áudio no Fone (TTS):** Motor de heurísticas com controle de *Debounce*, *Cooldown* e priorização contextual (placar/tempo de jogo) que sintetiza alertas curtos e diretos no fone do jogador sem gerar sobrecarga cognitiva.
* 📱 **Painel Tático em Tempo Real (Segunda Tela / Tablet):** Servidor `FastAPI + WebSockets` que transmite a telemetria espacial a 30 FPS para um segundo monitor, tablet ou smartphone, renderizando o campo 2D com malha de Voronoi, vetores de velocidade e linhas de passe em HTML5 Canvas, além de armazenar o histórico completo para analytics pós-jogo.

---

## 🛡️ Arquitetura Técnica & Segurança

O projeto foi desenhado sob dois pilares inegociáveis: **conformidade estrita com sistemas Anti-Cheat** e **determinismo de tempo real (< 15ms de latência ponta a ponta)**.

### 🔒 Garantia Anti-Cheat (*Zero Injection / 100% Passivo*)

> [!IMPORTANT]
> O **FC-Tactical-Engine** **NÃO injeta DLLs, NÃO lê nem altera a memória RAM do processo do jogo, NÃO realiza *hooking* de chamadas DirectX/Win32 e NÃO intercepta pacotes de rede ou executáveis do EA Sports FC / EA Javelin Anticheat**.

Toda a aquisição de dados ocorre de forma **100% externa e passiva** no nível do compositor de janela do sistema operacional, utilizando a API oficial **Windows Desktop Duplication API (DDA / DXGI)** — exatamente o mesmo mecanismo nativo utilizado por softwares legítimos de transmissão e gravação como **OBS Studio**, **Discord Screen Share** e **Nvidia ShadowPlay**. Para o jogo, o FC-Tactical-Engine é completamente invisível e indistinguível de uma captura de tela comum do Windows.

### ⚡ Pipeline de Ultrabaixa Latência

Para processar 60 quadros por segundo sem competir por recursos com o jogo (< 3% de CPU e < 5% de GPU), a arquitetura combina operações vetorizadas em C/C++ com desacoplamento assíncrono:

1. **Ingestão via GPU com `dxcam` (`< 4ms` por ciclo):** A captura acessa diretamente o *backbuffer* da placa de vídeo via DirectX Desktop Duplication API, extraindo apenas as Regiões de Interesse (*Multi-ROI*: radar, placar e HUD) já em memória e contornando a escala de DPI do Windows (`SetProcessDpiAwareness`).
2. **Desacoplamento Concorrente com Fila *Drop-Oldest* (LIFO tamanho 1):** O produtor de frames (GPU) e o consumidor (Visão Computacional) são isolados por um buffer concorrente de capacidade unitária (`maxlen=1`) com política *drop-oldest*. O motor tático **sempre processa o frame mais recente disponível**, garantindo **zero latência acumulada (*zero backpressure / frame lag*)** mesmo sob variações momentâneas de carga.
3. **Extração Colorimétrica Matricial com `OpenCV` (`< 3ms`):** Segmentação otimizada no espaço de cores HSV, filtragem morfológica e cálculo de momentos espaciais (`cv2.moments`) para localizar os 22 jogadores e a bola com suavização temporal exponencial contra *flickering*.
4. **Geometria Computacional com `SciPy`, `Shapely` e `NumPy` (`< 5ms`):** Construção vetorizada de Diagramas de Voronoi (`scipy.spatial.Voronoi`), interseção de polígonos táticos e álgebra linear em `NumPy` para avaliar linhas defensivas e vetores de passe.
5. **Persistência Colunar Analítica com `DuckDB`:** Gravação assíncrona em lotes (*bulk append*) em banco colunar OLAP embarcado (`DuckDB`), permitindo ingerir centenas de milhares de *ticks* de telemetria durante 90 minutos sem vazamento de memória e executar consultas analíticas pós-jogo em `< 50ms`.

### 🔀 Diagrama do Pipeline de Dados (End-to-End)

```text
+-----------------------------------------------------------------------------------+
|                      GPU FRAMEBUFFER (Windows DirectX / DXGI)                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                          [ Desktop Duplication API (DDA) ]
                          [  dxcam Multi-ROI Crop (< 4ms) ]
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|          FILA CONCORRENTE DROP-OLDEST (LIFO Size = 1 | Zero Latência)             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                CAMADA DE VISÃO COMPUTACIONAL (OpenCV / NumPy < 3ms)               |
|  +-----------------------------+               +-------------------------------+  |
|  | Game State & HUD Detector   |               | Radar Tracker (HSV + Moments) |  |
|  | (Em Jogo, Pausa, Replay)    |               | Centróides (X,Y) [0.0, 1.0]^2 |  |
|  +--------------+--------------+               +---------------+---------------+  |
+-----------------|----------------------------------------------|------------------+
                  +----------------------+-----------------------+
                                         |
                         [ DTO Canônico: FrameTelemetry ]
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               MOTOR TÁTICO ESPACIAL (SciPy / Shapely / NumPy < 5ms)               |
|   * Diagramas de Voronoi (Dominância Territorial)                                 |
|   * Compactação, Profundidade e Espaçamento da Linha de Zaga                      |
|   * Grafo de Passes Abertos & Polígonos de Sombra de Marcação                     |
+-----------------+-------------------------------+-------------------------------+-+
                  |                               |                               |
                  v                               v                               v
   +------------------------------+ +---------------------------+ +-------------------------------+
   |    COACH VIRTUAL (Áudio)     | |    SEGUNDA TELA (Web)     | |   TELEMETRIA OLAP (Storage)   |
   |  Debounce + Cooldown + TTS   | |  FastAPI + WebSockets     | |  Bulk Insert Assíncrono       |
   |  Alertas no Fone (< 100ms)   | |  Canvas 2D Real-Time 30Hz | |  DuckDB Colunar / Parquet     |
   +------------------------------+ +---------------------------+ +-------------------------------+
```

---

## 💻 Pré-requisitos de Sistema

Antes de iniciar a instalação, certifique-se de que o seu ambiente atende aos requisitos abaixo:

* **Sistema Operacional:** Windows 10 ou Windows 11 (64-bit) — *obrigatório devido à dependência nativa da API DirectX Desktop Duplication (DXGI)*.
* **GPU Dedicada:** Placa de vídeo com suporte a **DirectX 11 / DirectX 12** (AMD Radeon, NVIDIA GeForce ou Intel Arc) com drivers atualizados.
* **Python:** Versão **3.11** ou **3.12** (64-bit) instalada e adicionada ao `PATH` do sistema.
* **Microsoft Visual C++ Build Tools:** Necessário com a carga de trabalho **"Desenvolvimento para desktop com C++"** (*Desktop development with C++*) instalada para compilação e execução de extensões nativas e bindings COM/DirectX.

---

## 🚀 Guia de Instalação e Execução (PowerShell)

Siga o passo a passo abaixo diretamente no **Windows PowerShell** para configurar o ambiente local e validar seu hardware:

### 1. Clonar o repositório
```powershell
git clone https://github.com/trsthales/pc-tactical-engine.git
cd pc-tactical-engine
```

### 2. Criar e ativar o ambiente virtual (`.venv`)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> *Nota: Caso o PowerShell bloqueie a execução do script de ativação, execute `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` e tente novamente.*

### 3. Instalar as dependências do projeto
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar o diagnóstico de sanidade de hardware e dependências
O repositório inclui o utilitário [`sanity_check.py`](sanity_check.py), que verifica automaticamente todas as bibliotecas críticas, enumera sua GPU e monitores via DirectX DDA (`dxcam`) e testa o motor colunar `DuckDB` em memória:

```powershell
python sanity_check.py
```

**Saída esperada ao final do diagnóstico:**
```text
=================================================================
STATUS FINAL: [100% OPERACIONAL] Ambiente pronto para desenvolvimento!
=================================================================
```

### 5. Rodar o motor tático com a segunda tela
```powershell
python main.py
```
Abre o painel em `http://localhost:8000` e mostra no terminal o endereço para abrir no tablet ou celular (mesma rede Wi-Fi; o Windows pede liberação no firewall na primeira vez). Opções: `--no-audio`, `--no-db`, `--no-browser`, `--host 127.0.0.1` (só este PC), `--port`.

> [!NOTE]
> Enquanto a visão computacional (Marco 2) não fica pronta, a fonte de dados é um **radar simulado** (`src/simulation/`). Ele já exercita o motor tático, os alertas de voz, a gravação e o painel de ponta a ponta.

### 6. Ferramentas
```powershell
python -m tools.grab_samples --every 3 --count 20   # prints do jogo em tests/samples/ (insumo do Marco 2)
python -m tools.calibrate_roi tests/samples/<print>.png   # ajusta radar/placar/HUD -> config/rois.json
python -m src.capture.screen_capture --seconds 10   # benchmark da captura (FPS e ms por frame)
python -m src.storage.post_match_reports --export parquet   # relatório da última partida gravada
```

### 7. Testes e qualidade
```powershell
pytest
ruff check . ; ruff format --check config src tools tests main.py
mypy config src tools tests main.py
```

---

## 📂 Estrutura do Repositório (`src-layout`)

O projeto adota uma arquitetura modular limpa baseada em separação estrita de responsabilidades:

```text
FC-Tactical-Engine/
├── config/                 # Configurações globais, resoluções de referência e coordenadas de ROIs (Pydantic/JSON)
│   └── settings.py
├── data/                   # Armazenamento local de bancos DuckDB (.duckdb) e exportações analíticas (.parquet/.csv)
├── docs/                   # Documentação arquitetural, especificações técnicas e roadmap evolutivo
│   └── roadmap.md
├── src/                    # Código-fonte principal do motor tático
│   ├── capture/            # Captura de tela via GPU (dxcam/DDA), consciência de DPI Win32 e fila Drop-Oldest
│   ├── vision/             # Processamento OpenCV (HSV, contornos, momentos), OCR do placar e máquina de estados
│   ├── tactics/            # Geometria computacional (SciPy Voronoi, linhas defensivas, vetores de passe e regras)
│   ├── models/             # Contratos de dados imutáveis, DTOs e esquemas de validação (Pydantic)
│   ├── storage/            # Persistência telemétrica colunar assíncrona no DuckDB e relatórios pós-jogo
│   └── alerts/             # "Coach Virtual": motor de heurísticas (debounce/cooldown) e fila assíncrona TTS
├── tests/                  # Suíte de testes automatizados (pytest) e amostras visuais de referência
│   └── samples/
├── requirements.txt        # Dependências fixadas para captura, visão, geometria, banco colunar e servidor web
├── sanity_check.py         # Script de diagnóstico de ambiente, GPU DirectX DDA e DuckDB in-memory
└── README.md               # Documentação principal do projeto
```

* **`src/capture/`**: Responsável pela interface de baixo nível com a GPU via DirectX Desktop Duplication API (`dxcam`), neutralização de escala de DPI do Windows e gerenciamento de threads de captura com fila *Drop-Oldest*.
* **`src/vision/`**: Pipeline determinístico de visão computacional em `OpenCV` para segmentar jogadores e bola no radar 2D, extrair centróides normalizados e identificar o estado atual da partida.
* **`src/tactics/`**: Núcleo matemático que consome as coordenadas normalizadas e calcula Diagramas de Voronoi, métricas de compactação, linhas de impedimento/zaga e disponibilidade de linhas de passe.
* **`src/models/`**: Define as entidades de domínio e *Data Transfer Objects* (DTOs) tipados que trafegam entre captura, visão, motor tático, banco de dados e WebSockets.
* **`src/storage/`**: Camada de persistência de alta vazão sobre `DuckDB` para gravação em lote (*bulk insert*) dos frames da partida e execução de queries analíticas SQL.
* **`src/alerts/`**: Gerenciador de eventos táticos com filtros de *cooldown* contextual e despacho de síntese de voz (TTS) assíncrona para orientar o jogador sem atrasos.
* **`config/`**: Centraliza parâmetros de resolução, taxas de amostragem (FPS), limiares táticos e calibração de Regiões de Interesse (*Multi-ROI*).
* **`tests/`**: Testes unitários e de integração focados em invariância de resolução (720p a 4K), performance por ciclo e precisão geométrica, acompanhados de *frames* reais em `tests/samples/`.

---

## 🗺️ Roadmap & Status do Projeto

O desenvolvimento do **FC-Tactical-Engine** é guiado por marcos incrementais com critérios de aceite (*Definition of Done*) focados em latência e determinismo.

| Marco | Módulo / Entrega Principal | Latência Alvo / Meta | Status |
| :--- | :--- | :---: | :---: |
| **Marco 0** | **Fundação, Arquitetura `src-layout` & Diagnóstico de Hardware (`sanity_check.py`)** | Setup Validado | ✅ **Concluído** |
| **Marco 1** | **Camada de Ingestão Visual (`dxcam` 60 FPS, Win32 DPI & Multi-ROI Dinâmico)** | `< 4ms` / frame | 🧪 **Implementado** (validar no Windows) |
| **Marco 2** | **Visão Computacional & Rastreamento de Entidades (Radar HSV + Estado do Jogo)** | `< 3ms` / frame | ⏳ Planejado |
| **Marco 3** | **Motor Tático Espacial (Diagramas de Voronoi, Linhas de Zaga & Rede de Passes)** | `< 5ms` / ciclo | ✅ **Implementado** |
| **Marco 4** | **"Coach Virtual" (Motor de Heurísticas, Debounce/Cooldown & Áudio TTS)** | `< 100ms` disparo | 🧪 **Implementado** (voz e latência a validar no Windows) |
| **Marco 5** | **Ingestão Telemétrica Colunar (`DuckDB` Bulk Insert) & Analytics Pós-Jogo** | `< 50ms` query | ✅ **Implementado** |
| **Marco 6** | **Segunda Tela em Tempo Real (`FastAPI` + `WebSockets` + Canvas 2D) & Orquestração** | `< 15ms` end-to-end | 🚧 **Parcial**: painel e orquestração com o simulador; captura não integrada, metas de latência e CPU não medidas |

📖 Para consultar o detalhamento técnico completo de cada entregável e seus respectivos critérios de aceite, acesse o documento de planejamento: **[`docs/roadmap.md`](docs/roadmap.md)**.

---

## 🤝 Contribuição & Licença

Contribuições são muito bem-vindas! Se você tem interesse em **Visão Computacional de Baixa Latência**, **Geometria Computacional Aplicada ao Futebol** ou **Engenharia de Telemetria em Tempo Real**:

1. Faça um *Fork* deste repositório.
2. Crie uma *branch* para sua feature (`git checkout -b feat/minha-feature`).
3. Certifique-se de que o diagnóstico (`python sanity_check.py`) e a suíte de testes (`pytest`) passam sem erros.
4. Envie um *Pull Request* detalhando o impacto de latência e os testes adicionados.

Distribuído sob a licença **MIT**. Consulte o cabeçalho do repositório para mais informações.
