# 🗺️ Roadmap Arquitetural — FC-Tactical-Engine

> Planejamento evolutivo dos marcos de engenharia do **FC-Tactical-Engine**, desde a fundação de captura de baixa latência via GPU até a segunda tela interativa em tempo real e analytics pós-jogo.

```text
[ Marco 0 ] Setup & Ambiente (CONCLUÍDO ✅)
     │
     ▼
[ Marco 1 ] Camada de Ingestão de Baixa Latência & Multi-ROI Dinâmico (EM ANDAMENTO 🚧)
     │
     ▼
[ Marco 2 ] Visão Computacional & Rastreamento das Entidades (Radar + HUD) (IMPLEMENTADO ✅)
     │
     ▼
[ Marco 3 ] Motor Tático Espacial (Voronoi, Linhas de Zaga & Passes)
     │
     ▼
[ Marco 4 ] "Coach Virtual" (Sintetizador de Áudio & Alertas Contextuais)
     │
     ▼
[ Marco 5 ] Ingestão Telemétrica Colunar & Analytics Pós-Jogo (DuckDB)
     │
     ▼
[ Marco 6 ] Segunda Tela em Tempo Real (WebSockets + Canvas) & Validação End-to-End
```

---

## ✅ Marco 0: Fundação & Ambiente de Desenvolvimento *(CONCLUÍDO)*
* [x] Estrutura modular `src-layout` (`src/`, `config/`, `tests/`, `data/`, `docs/`) estabelecida.
* [x] Configuração do ambiente virtual (`.venv`) e instalação de dependências nativas (`dxcam`, `opencv-python`, `scipy`, `shapely`, `duckdb`, `pydantic`, `pyttsx3`, `fastapi`, `uvicorn`, `websockets`, `pytest`).
* [x] Validação de GPU (Radeon RX 550 / DirectX 11/12), monitor principal e DirectX Desktop Duplication API (DDA) via `sanity_check.py`.
* [x] Conexão e query analítica `DuckDB` em memória (`:memory:`) operacionais.

---

## 🚧 Marco 1: Camada de Ingestão Visual & Multi-ROI Dinâmico *(EM ANDAMENTO)*
> **Objetivo:** Capturar o fluxo do jogo da GPU a 60 FPS com latência mínima, desacoplar os threads e extrair recortes normalizados e imunes a mudanças de resolução ou DPI do Windows.

* **Entregáveis Técnicos:**
  1. `src/capture/win32_dpi.py`: Inicialização com chamada Win32 (`SetProcessDpiAwareness(2)`) para neutralizar a escala de DPI do Windows.
  2. `config/settings.py`: Modelagem Pydantic de múltiplos ROIs normalizados (`radar`, `scoreboard`, `player_hud`) variando de `0.0` a `1.0`.
  3. `src/capture/screen_capture.py`: Serviço assíncrono em thread dedicada usando `dxcam` com fila concorrente *drop-oldest* (LIFO de tamanho = 1).
  4. `tools/calibrate_roi.py`: Interface OpenCV com sliders interativos para ajuste visual e salvamento das coordenadas no `config/radar_roi.json`.
* **Critérios de Aceite (DoD):**
  * Loop de captura sustentando 60 FPS com latência de extração `< 4ms`.
  * Teste unitário aprovado comprovando que os mesmos recortes funcionam identicamente em 720p, 1080p, 1440p e 4K.

---

## ✅ Marco 2: Visão Computacional & Rastreamento de Entidades *(IMPLEMENTADO)*
> **Objetivo:** Converter os pixels dos recortes em telemetria numérica utilizável (máquina de estados, centróides dos 22 jogadores + bola e leitura de tempo/placar).

> **Como ficou** (validado num vídeo de partida inteira do EA FC 27, 1080p, 58.785 frames):
> * `radar_geometry.py`: homografia das 4 bordas do trapézio do radar (perspectiva) para `[0, 1]²`; tudo calibrado de um fundo médio (`tools/calibrate_radar.py`).
> * `radar_markers.py`: HSV + componentes conexos com área esperada por profundidade; separa marcadores fundidos (círculos pela área, triângulos pela forma); aceita a ponta de triângulo coberta por um círculo; descarta cena sem a borda do sprite.
> * `radar_tracker.py`: identidade estável (algoritmo húngaro + portão físico), filtro alfa-beta, jogador escondido atrás de outro marcador segue quem o cobre.
> * `ocr_hud.py`: placar e relógio por modelos de dígito (1.702 de 1.702 leituras certas no vídeo, nenhuma recusada); menu de pausa pela cor.
> * `game_state_detector.py`: estado pela interface (radar, placar, menu) e lado de ataque pelos goleiros.
> * Medido: 1,6 ms/frame no pipeline inteiro (radar < 3 ms); 0 voltas do relógio; 9 gols, todos +1; lado de ataque troca 1 vez (intervalo); 11 x 11 rastreados em 72% do jogo rolando (o resto é marcador escondido de verdade: atrás de outro, do árbitro ou do cinegrafista).
> * Falta: validar ao vivo no Windows (`python main.py --screen`).

* **Entregáveis Técnicos:**
  1. `src/vision/game_state_detector.py`: Classificador leve (via histogramas/templates) para detectar o estado da partida: `EM_ANDAMENTO`, `BOLA_PARADA`, `REPLAY` ou `PAUSA`.
  2. `src/vision/radar_tracker.py`:
     * Segmentação por espaço de cores HSV das três classes: Time A, Time B e Bola.
     * Extração de centróides `(X, Y)` via contornos e momentos espaciais do OpenCV.
     * Filtro temporal básico (suavização exponencial) para eliminar *flickering* (piscamento) de pontos no radar.
  3. `src/vision/ocr_hud.py`: Leitura rápida dos minutos e do placar na região do placar.
* **Critérios de Aceite (DoD):**
  * Módulo processando cada frame do radar em tempo `< 3ms`.
  * Extração consistente de coordenadas no plano unitário `[0.0, 1.0]²`.
  * Máquina de estados silenciando o fluxo imediatamente ao entrar em replay ou pausa.

---

## ✅ Marco 3: Motor Tático Espacial (Geometria Computacional) *(IMPLEMENTADO)*
> **Objetivo:** Transformar as coordenadas brutas dos jogadores em métricas analíticas e inteligência espacial de futebol em tempo real.

* **Entregáveis Técnicos:**
  1. `src/tactics/spatial_engine.py`:
     * **Diagramas de Voronoi:** Particionamento do campo com `scipy.spatial` para medir percentual de controle territorial.
     * **Análise de Linhas:** Cálculo vetorial da distância entre zaga e ataque (profundidade) e espaçamento entre zagueiros centrais.
     * **Rede de Passes:** Traçado de vetores entre o portador da bola e receptores elegíveis, calculando polígonos de interceptação de marcadores adversários.
  2. `src/tactics/event_rules.py`: Motor de regras baseado em limiares configuráveis (ex: `distancia_zagueiros > threshold` -> evento `CENTRO_EXPOSTO`).
* **Critérios de Aceite (DoD):**
  * Tempo total de cálculo de Voronoi + linhas de passe `< 5ms` por ciclo.
  * Emissão de DTOs estruturados com os eventos táticos identificados a cada tick.

---

## 🧪 Marco 4: O "Coach Virtual" (Motor de Alertas e Áudio) *(IMPLEMENTADO, validar voz no Windows)*
> **Objetivo:** Transformar os eventos táticos em comandos audíveis claros e úteis no fone do jogador, sem causar sobrecarga cognitiva.

* **Entregáveis Técnicos:**
  1. `src/alerts/audio_dispatcher.py`: Fila assíncrona de sintetização de voz com o `pyttsx3` ou `edge-tts`.
  2. `src/alerts/heuristics_manager.py`:
     * Sistema de **Debounce e Cooldown:** Impede que o mesmo alerta seja repetido em uma janela de 10 a 15 segundos.
     * **Priorização Contextual:** Filtra alertas com base no placar e no tempo (ex: aos 85' vencendo, suprime alertas de ataque arriscado).
* **Critérios de Aceite (DoD):**
  * Alerta de voz disparado em menos de `100ms` a partir da detecção do evento.
  * Zero travamento ou engasgo no áudio mesmo sob volume alto de eventos.

---

## ✅ Marco 5: Ingestão Telemétrica Colunar & Analytics Pós-Jogo *(IMPLEMENTADO)*
> **Objetivo:** Gravar o histórico completo da partida em disco para auditoria, relatórios no intervalo e análises profundas pós-jogo.

* **Entregáveis Técnicos:**
  1. `src/storage/telemetry_db.py`:
     * Conexão com **DuckDB** local.
     * Esquema colunar particionado por `match_id` (`tick_id`, `timestamp`, `entity_id`, `team`, `x`, `y`, `vx`, `vy`).
     * Thread dedicada com escrita em lotes (*bulk insert*) a cada 1 ou 2 segundos para evitar contenção de disco.
  2. `src/storage/post_match_reports.py`: Scripts SQL analíticos para geração de mapas de calor, estatísticas de posse e exportação para `.csv` ou `.parquet`.
* **Critérios de Aceite (DoD):**
  * Gravação de 90 minutos simulados de partida sem consumo de memória crescente (zero leak).
  * Execução de consultas analíticas em menos de `50ms` sobre centenas de milhares de linhas no DuckDB.

---

## 🚧 Marco 6: Segunda Tela (WebSockets) & Integração End-to-End *(PARCIAL: sem captura real)*
> **Objetivo:** Unificar todos os módulos no script principal e disponibilizar um painel visual em tempo real para monitor secundário ou celular.

* **Entregáveis Técnicos:**
  1. `src/server/app.py`: Servidor `FastAPI` com endpoint `WebSocket` transmitindo o estado canônico do jogo a 30 FPS.
  2. `src/server/static/`: Interface web minimalista em HTML5 Canvas que renderiza o campo 2D espelhado com os polígonos de Voronoi e as linhas de passe.
  3. `main.py`: Ponto de entrada (*orchestrator*) com orquestração de threads, métricas internas de performance (FPS, tempos de ciclo) e *graceful shutdown*.
* **Critérios de Aceite (DoD):**
  * Sistema completo rodando concomitantemente com o jogo consumindo menos de 3% de CPU e 5% de GPU.
  * Latência end-to-end (da captura da tela até o áudio/segunda tela) inferior a **15 milissegundos**.
