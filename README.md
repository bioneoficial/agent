# Git Terminal Assistant (GTA) - TCC

GTA é um assistente inteligente de terminal que utiliza uma arquitetura multi-agente para executar operações Git e criar/editar arquivos de código. O sistema é focado especialmente em gerar mensagens de commit semânticas seguindo o padrão Conventional Commits.

## Arquitetura

O sistema utiliza uma arquitetura de agentes especializados com um poderoso sistema de workflow híbrido que combina execução direta com planejamento dinâmico. A arquitetura foi projetada para ser robusta, flexível e auto-corretiva.

### Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Workflow System                        │
│                                                                 │
│  ┌─────────────┐    ┌───────────────────────────────────────┐   │
│  │  Git Agent  │    │            Code Agent                 │   │
│  │             │    │                                       │   │
│  │ • Git cmds  │    │ • File operations (create/edit/read)  │   │
│  │ • Commit    │    │ • Test execution & generation         │   │
│  │ • Status    │    │ • Code analysis & refactoring         │   │
│  └─────────────┘    │ • Project structure management        │   │
│         │           └───────────────────────────────────────┘   │
│         │                                                      │
│         │           ┌──────────────────┐                       │
│         └───────────┤   Chat Agent     │                       │
│                     │                  │                       │
│                     │ • General Q&A    │                       │
│                     │ • Documentation  │                       │
│                     └──────────────────┘                       │
│                             │                                 │
│                    ┌────────┴────────┐                        │
│                    │   Orchestrator   │                        │
│                    │                  │                        │
│                    │ • Route requests │                        │
│                    │ • Handle errors  │                        │
│                    └────────┬────────┘                        │
│                             │                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              Workflow Execution Engine                │    │
│  │                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │    │
│  │  │  Task       │  │  Validation │  │  Retry &     │  │    │
│  │  │  Execution  │◄─┤  & Error    │◄─┤  Replanning  │  │    │
│  │  └─────────────┘  │  Handling   │  │  System      │  │    │
│  │         ▲         └─────────────┘  └───────┬───────┘  │    │
│  │         │                    ▲              │          │    │
│  │         └────────────────────┴──────────────┘          │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Principais

1. **Agentes Especializados**
   - **GitAgent**: Gerencia operações Git e geração de mensagens de commit semânticas
   - **CodeAgent**: Executa operações de arquivo, testes e análise de código
   - **ChatAgent**: Fornece respostas a perguntas gerais e documentação

2. **Sistema de Workflow Híbrido**
   - **Motor de Execução**: Gerencia o fluxo de trabalho e orquestra tarefas
   - **Validação Automática**: Verifica a validade do código gerado
   - **Sistema de Repetição Inteligente**: Tenta novamente tarefas com falha com contexto aprimorado
   - **Replanejamento Dinâmico**: Ajusta o fluxo de trabalho com base nos resultados

3. **Gerenciamento de Estado**
   - Rastreia o progresso das execuções
   - Mantém histórico de tentativas e erros
   - Fornece contexto para decisões de replanejamento

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

### 2. **CodeAgent** - Especialista em Código (Consolidado + IA Avançada)
- **Operações de Arquivo**
  - Criação, edição e leitura de arquivos
  - Suporte a múltiplas linguagens de programação
  - Backup automático de arquivos durante edições
  
- **Validação Automática de Código** 🆕
  - Validação automática de sintaxe Python em tempo real
  - Verificação de imports e dependências
  - Correção automática de erros usando LLM
  - Integração transparente no fluxo de geração de código
  
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

## Sistema de Raciocínio Estruturado (Chain-of-Thought)

O GTA inclui um sistema avançado de raciocínio estruturado que permite planejamento multi-etapas com análise contextual e rastreabilidade completa.

### Características do Sistema CoT

- **Planejamento Estruturado**: Decomposição automática de tarefas complexas em etapas executáveis
- **Validação JSON**: Esquemas Pydantic para garantir consistência e validação de dados
- **Rastreamento de Execução**: Logs detalhados de cada etapa com pré/pós-condições
- **Análise de Riscos**: Identificação proativa de possíveis problemas e mitigações
- **Recuperação de Erros**: Estratégias de fallback e reexecução inteligente

## Sistema de Workflow Híbrido

O sistema de workflow híbrido combina a simplicidade do LangGraph com recursos avançados de validação, repetição e replanejamento para execução confiável de tarefas complexas.

### Recursos Principais

- **Validação Automática de Código**
  - Verificação de sintaxe Python
  - Validação de importações
  - Correção automática de erros comuns

- **Repetição Inteligente**
  - Análise de falhas para determinar a causa raiz
  - Aprimoramento de contexto para novas tentativas
  - Limites configuráveis de repetição

- **Replanejamento Dinâmico**
  - Adapta o fluxo de trabalho com base nos resultados
  - Toma decisões baseadas em confiança
  - Preserva o contexto entre tentativas

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `GTA_WORKFLOW_MAX_RETRIES` | 3 | Número máximo de tentativas por tarefa |
| `GTA_WORKFLOW_CONFIDENCE_THRESHOLD` | 0.8 | Limiar de confiança para aceitar resultados |
| `GTA_CODE_VALIDATION_ENABLED` | true | Habilita validação de código |
| `GTA_CODE_VALIDATION_STRICT` | false | Se verdadeiro, falha em erros de validação |
| `GTA_CODE_AUTO_CORRECTION` | true | Tenta corrigir automaticamente erros de código |
| `GTA_RETRY_LEARNING_ENABLED` | true | Aprende com tentativas anteriores |
| `GTA_RETRY_CONTEXT_ENHANCEMENT` | true | Melhora o contexto em novas tentativas |

### Documentação Detalhada

Consulte os guias detalhados para obter mais informações:

- [Guia de Uso](./docs/USAGE_GUIDE.md) - Exemplos práticos e instruções
- [Sistema de Workflow Híbrido](./docs/HYBRID_WORKFLOW_SYSTEM.md) - Documentação técnica detalhada
- [Provedores LLM](./docs/LLM_PROVIDERS.md) - Configuração e suporte a múltiplos modelos

### Modos de Raciocínio

Configuráveis via `GTA_REASONING_MODE`:

1. **`none`** - Sem raciocínio estruturado, apenas saída final
2. **`brief`** - Plano simplificado com 3-7 etapas principais  
3. **`structured`** - Trace JSON completo com validação (recomendado)

### Armazenamento de Traces

Os traces de raciocínio são salvos em `.orchestra/runs/` com:
- Metadados da execução
- Logs detalhados por etapa
- Contexto e decisões tomadas
- Métricas de performance

## Sistema Híbrido de Workflow com IA Avançada

O GTA agora inclui um sistema híbrido de workflow que combina a simplicidade do LangGraph com capacidades avançadas de planejamento dinâmico, validação automática e recuperação inteligente de erros.

### Funcionalidades do Sistema Híbrido

- **Planejamento Dinâmico**: Replanning automático baseado em resultados de execução
- **Retry Inteligente**: Sistema de retry que aprende com falhas anteriores
- **Validação Automática**: Validação de código em tempo real com correção automática
- **Saída Estruturada**: Modelos Pydantic para garantir consistência de dados
- **Análise de Confiança**: Decisões baseadas em scores de confiança
- **Feedback de Erro**: Categorização inteligente de erros com sugestões de correção

### Modelos de Dados Estruturados

O sistema utiliza modelos Pydantic robustos para estruturar todos os dados:

- `TaskResult` - Resultados estruturados de execução de tarefas
- `ValidationResult` - Feedback detalhado de validação
- `ErrorFeedback` - Análise categorizada de erros
- `ReplanDecision` - Decisões inteligentes de replanning
- `WorkflowResult` - Resultados de workflow completo
- `TaskMetadata` - Metadados de execução

### Retry Inteligente

O sistema de retry analisa falhas anteriores e ajusta a abordagem:

```python
# Exemplo de retry com contexto aprimorado
if "syntax" in error_message:
    context["auto_correct"] = True
    context["validation_strict"] = True
elif "import" in error_message:
    context["check_dependencies"] = True
```

## Sistema de Percepção Proativa

O sistema de percepção monitora continuamente o projeto e gera sugestões contextuais inteligentes.

### Monitoramento Automático

- **Filesystem Watcher**: Detecta mudanças em arquivos e diretórios
- **Git Watcher**: Monitora status do repositório, commits e conflitos
- **Análise Contextual**: Identifica padrões e oportunidades de melhoria

### Tipos de Sugestões

1. **Qualidade de Código**: Refatoração, duplicação, padrões de design
2. **Workflow Git**: Commits grandes, conflitos, branching
3. **Testes**: Cobertura, automação, frameworks de teste
4. **Documentação**: README, comentários, APIs
5. **Segurança**: Arquivos sensíveis, secrets, permissões

### Interação com Sugestões

```bash
# Listar sugestões ativas
gta> list

# Aceitar uma sugestão
gta> accept perception_1_1234567890

# Rejeitar uma sugestão  
gta> dismiss perception_2_1234567891

# Ver detalhes
gta> show perception_1_1234567890

# Ajuda com comandos
gta> help
```

### Configuração da Percepção

```bash
# Ativar/desativar sistema
GTA_PERCEPTION_ENABLED=1

# Intervalo entre sugestões (segundos)
GTA_PERCEPTION_COOLDOWN=300

# Modo silencioso
GTA_PERCEPTION_SILENT=0

# Máximo de sugestões simultâneas
GTA_PERCEPTION_MAX_CONCURRENT=3
```

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

## Vantagens da Arquitetura Multi-Agente Híbrida

### Core Benefits
1. **Especialização**: Cada agente é otimizado para sua tarefa
2. **Manutenibilidade**: Código modular e fácil de debugar
3. **Extensibilidade**: Novos agentes podem ser adicionados facilmente
4. **Confiabilidade**: Sanitização automática de respostas do LLM
5. **Performance**: Apenas o agente necessário é ativado

### Benefícios do Sistema Híbrido 🆕
6. **Auto-Correção**: Validação e correção automática de código
7. **Resiliência**: Retry inteligente com análise de falhas
8. **Observabilidade**: Saída estruturada com metadados detalhados
9. **Adaptabilidade**: Replanning dinâmico baseado em contexto
10. **Confiança**: Decisões baseadas em scores de confiança
11. **Rastreabilidade**: Histórico completo de execução e decisões

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
│   ├── base_agent.py        # Classe base com sanitização
│   ├── git_agent.py         # Operações Git
│   ├── code_agent.py        # Criação/edição/validação de código
│   ├── workflow_executor.py # Sistema híbrido de workflow 🆕
│   └── orchestrator.py      # Roteador de requisições
├── orchestra/
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── reasoning.py     # Esquemas de raciocínio CoT
│   │   └── task_results.py  # Modelos Pydantic estruturados 🆕
│   ├── perception/          # Sistema de percepção
│   └── utils/               # Utilitários do sistema
├── main.py                  # Entry point
├── llm_backend.py           # Configuração do LLM
├── llm_providers.py         # Fábrica de provedores LLM
├── install.sh               # Instalador (macOS/Linux)
├── gta                      # Script global (macOS/Linux)
├── gta.ps1                  # Script PowerShell (Windows)
├── gta.cmd                  # Script cmd.exe (Windows)
├── docs/
│   └── LLM_PROVIDERS.md     # Guia de provedores LLM e exemplos
└── README.md                # Este arquivo
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