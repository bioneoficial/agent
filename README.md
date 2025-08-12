# Git Terminal Assistant (GTA) - TCC

GTA é um assistente inteligente de terminal que utiliza uma arquitetura multi-agente para executar operações Git e criar/editar arquivos de código. O sistema é focado especialmente em gerar mensagens de commit semânticas seguindo o padrão Conventional Commits.

## Arquitetura

O sistema utiliza uma arquitetura de agentes especializados, otimizada para eficiência e manutenibilidade. A arquitetura atual foi simplificada para três agentes principais:

```
┌─────────────────┐    ┌───────────────────────────────────────┐
│   Git Agent     │    │            Code Agent                 │
│                 │    │                                       │
│ • Git commands  │    │ • File operations (create/edit/read) │
│ • Commit msgs   │    │ • Test execution & generation       │
│ • Status/diff   │    │ • Code analysis & refactoring      │
└─────────────────┘    │ • Project structure management     │
         │             └───────────────────────────────────────┘
         │
         │             ┌──────────────────┐
         └─────────────┤   Chat Agent     │
                       │                  │
                       │ • General Q&A    │
                       │ • Documentation  │
                       └──────────────────┘
                               │
                      ┌────────┴────────┐
                      │   Orchestrator   │
                      │                  │
                      │ • Route requests │
                      │ • Handle errors  │
                      └──────────────────┘
```

### Agentes Principais

1. **GitAgent**
   - Comandos e operações Git
   - Mensagens de commit semânticas
   - Gerenciamento de branches e repositórios

2. **CodeAgent** (Consolidado)
   - Operações de arquivo (criar/editar/ler)
   - Execução e geração de testes
   - Análise e refatoração de código
   - Gerenciamento de estrutura de projetos

3. **ChatAgent**
   - Respostas a perguntas gerais
   - Documentação e ajuda
   - Suporte a tarefas diversas

## Características Principais

### 1. **GitAgent** - Especialista em Versionamento
- Gera mensagens de commit inteligentes seguindo Conventional Commits
- Executa comandos git com segurança
- Analisa diffs e status do repositório
- Categoriza alterações de arquivos para determinar o tipo de commit (feat, fix, etc.)
- Formata automaticamente mensagens seguindo padrões de commits convencionais
- Gerencia branches e operações de repositório remoto

### 2. **CodeAgent** - Especialista em Código (Consolidado)
- **Operações de Arquivo**
  - Criação, edição e leitura de arquivos
  - Suporte a múltiplas linguagens de programação
  - Backup automático de arquivos durante edições
  
- **Testes**
  - Execução de testes unitários e de integração
  - Geração de testes automatizados
  - Análise de cobertura de testes
  
- **Análise e Refatoração**
  - Análise estática de código
  - Sugestões de refatoração
  - Verificação de estilo e boas práticas
  
- **Gerenciamento de Projetos**
  - Criação de estrutura de projetos
  - Visualização de hierarquia de arquivos
  - Gerenciamento de dependências

### 3. **ChatAgent** - Assistente de Desenvolvimento
- Respostas a perguntas técnicas
- Explicação de conceitos de programação
- Ajuda com documentação e boas práticas
- Suporte a tarefas gerais de desenvolvimento

## Instalação

Esta seção cobre Linux, macOS e Windows. Veja também `docs/LLM_PROVIDERS.md` para configurar provedores LLM (OpenAI, Anthropic, Google, Cohere, Azure, Ollama).

1) Clonar o repositório

```bash
git clone <repository-url>
cd tcc  # ou o diretório raiz do projeto
```

2) Criar ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# No Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

3) Instalar dependências

Opção B (recomendada) — requisitos divididos:

```bash
pip install -r requirements.txt                  # núcleo
pip install -r requirements-providers.txt        # provedores online (se necessário)
# ou tudo de uma vez:
pip install -r requirements-all.txt
```

4) Configurar LLM e Router

```bash
cp .env.example .env
# Edite .env e defina LLM_PROVIDER/LLM_MODEL e chaves de API, se aplicável
# Para habilitar roteamento por LLM:
# GTA_ROUTER=llm
# GTA_ROUTER_THRESHOLD=0.7
```

5) Instalar o comando global (macOS/Linux)

```bash
./install.sh
# Recarregue o shell (ex.: source ~/.zshrc) e use: gta -i
```

5b) Windows (PowerShell)

Você pode executar direto do repositório:

```powershell
./gta.ps1 -i
```

Se preferir o cmd.exe, você pode usar também o wrapper `gta.cmd`:

```bat
gta.cmd -i
```

Observação sobre Execution Policy (Windows):

Se você receber um erro como "running scripts is disabled on this system":

```powershell
# Permitir scripts para o usuário atual (recomendado)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# OU apenas desbloquear este arquivo específico
Unblock-File .\gta.ps1
```

Em ambientes corporativos, a política pode ser gerenciada pelo TI. Caso necessário, execute o PowerShell como Administrador ou consulte sua equipe de TI.

Para tornar global, adicione este atalho ao seu $PROFILE:

```powershell
# Abra o perfil
if (!(Test-Path $PROFILE)) { New-Item -Type File -Path $PROFILE -Force | Out-Null }
notepad $PROFILE

# Adicione esta função ao final do arquivo de perfil e ajuste o caminho do script
function gta { & "C:\caminho\para\o\projeto\gta.ps1" @args }
```

## Uso

### Modo Interativo
```bash
gta -i               # macOS/Linux após install.sh
# Windows: se sem atalho global, use:
# .\gta.ps1 -i  ou  .\gta.cmd -i
```

### Comando Único
```bash
gta "criar arquivo calculator.py com funções matemáticas"
gta "gerar testes para calculator.py"
gta "commit com mensagem descritiva"
```

### Comandos Específicos do GTA

```bash
# Operações Git
gta "commit com mensagem descritiva"
gta "criar branch feature/nova-funcionalidade"

# Operações de Código
gta "criar arquivo utils.py com funções úteis"
gta "executar testes em test_meucodigo.py"

# Análise de Código
gta "analisar complexidade ciclomática"
gta "sugerir melhorias de performance"

# Gerenciamento de Projeto
gta "mostrar estrutura do projeto"
gta "criar estrutura de projeto Python"


## Pipelines de Colaboração (multi-agente)

Os pipelines permitem que múltiplos agentes cooperem em um fluxo único, compartilhando contexto (ex.: resultados de testes) para ações mais inteligentes.

- commit_with_tests: executa testes via CodeAgent, gera mensagem de commit via GitAgent incluindo o resultado dos testes e, por padrão, só commita se os testes passarem.
- message_with_tests: executa testes e gera somente a mensagem de commit (sem commitar).

Exemplos:
```bash
gta "commit with tests"
gta "message with tests"
```

Configuração:
- Por padrão, commits são bloqueados quando há falhas nos testes.
- Para permitir commit mesmo com falhas, ajuste a variável de ambiente GTA_COMMIT_REQUIRE_TESTS_PASS:

macOS/Linux (bash/zsh):
```bash
export GTA_COMMIT_REQUIRE_TESTS_PASS=0
```

Windows PowerShell:
```powershell
$env:GTA_COMMIT_REQUIRE_TESTS_PASS="0"
```

Observações:
- A detecção é baseada em palavras‑chave ("test/tests/teste", "commit/commitar", "message/mensagem/msg").
- É necessário ter alterações staged para a geração de mensagem de commit.

### Comandos de Terminal (nativos)

O GTA executa diretamente diversos comandos de terminal comuns antes de encaminhar aos agentes. A disponibilidade pode variar por sistema operacional.

- Navegação e listagem: `ls`, `pwd`, `cd`, `mkdir`, `find`
- Terminal e sessão: `clear`, `history`, `alias`
- Processos e sistema: `ps`, `top`, `htop`
- Rede: `ping`, `ssh`, `curl`, `wget`
- Arquivos e arquivamento: `cat`, `grep`, `head`, `tail`, `tar`, `zip`, `unzip`
- Informações: `man`, `date`, `whoami`, `uname`, `df`, `du`

Exemplos:
```bash
ls -la
pwd
clear
```

## Comandos Especiais

- `help` - Mostra comandos disponíveis
- `agents` - Lista agentes e suas capacidades
- `exit` - Sai do assistente

## Vantagens da Arquitetura Multi-Agente

1. **Especialização**: Cada agente é otimizado para sua tarefa
2. **Manutenibilidade**: Código modular e fácil de debugar
3. **Extensibilidade**: Novos agentes podem ser adicionados facilmente
4. **Confiabilidade**: Sanitização automática de respostas do LLM
5. **Performance**: Apenas o agente necessário é ativado

## Requisitos

- Python 3.8+ (recomendado 3.10+)
- Git instalado
- Provedor LLM:
  - Ollama (opcional) com modelo local, ex.: `llama3.1:8b`
  - ou provedores online (OpenAI/Anthropic/Google/Cohere/Azure) com chave de API
- Windows: PowerShell 5+ (recomendado PowerShell 7+)

## Contribuindo

Para adicionar um novo agente:

1. Crie um arquivo em `agents/`
2. Estenda a classe `BaseAgent`
3. Implemente `can_handle()` e `process()`
4. Adicione ao orquestrador

## Estrutura do Projeto

```
tcc/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Classe base com sanitização
│   ├── git_agent.py       # Operações Git
│   ├── code_agent.py      # Criação/edição de código
│   └── orchestrator.py    # Roteador de requisições
├── main.py                # Entry point
├── llm_backend.py         # Configuração do LLM
├── llm_providers.py       # Fábrica de provedores LLM
├── install.sh             # Instalador (macOS/Linux)
├── gta                    # Script global (macOS/Linux)
├── gta.ps1                # Script PowerShell (Windows)
├── gta.cmd                # Script cmd.exe (Windows)
├── docs/
│   └── LLM_PROVIDERS.md   # Guia de provedores LLM e exemplos
└── README.md              # Este arquivo
```

## Troubleshooting

### .env não é carregado
- macOS/Linux: o script `gta` resolve o caminho real (symlink-safe) e carrega `./.env`. Verifique se o arquivo está no diretório raiz do projeto.
- Windows: `gta.ps1` carrega `.env` do diretório do script. Confirme o caminho do script usado no atalho.

### Provider não funciona ou volta para Ollama
- Verifique se instalou as dependências do provedor (ex.: `pip install -r requirements-providers.txt`).
- Confirme variáveis no `.env`: `LLM_PROVIDER`, `LLM_MODEL`, e chave `*_API_KEY` correspondente.
- Remova poluição de ambiente (variáveis residuais do shell). Feche e reabra o terminal ou `unset` variáveis conflitantes.

### Dependência ausente
Mensagens como: `OpenAI provider requires: pip install langchain-openai` indicam falta de pacote. Instale e tente novamente.

### Router LLM
- Para habilitar: `GTA_ROUTER=llm` no `.env`.
- Ajuste confiança: `GTA_ROUTER_THRESHOLD=0.7` (padrão).
- Debug: `GTA_ROUTER_DEBUG=1` mostra estratégia e decisões de roteamento.

### Ollama local
Certifique-se de que o servidor está ativo e o modelo instalado:
```bash
ollama pull llama3.1:8b
ollama serve
```

## Licença

MIT

Desenvolvido com arquitetura multi-agente para máxima eficiência e manutenibilidade. 