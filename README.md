# Git Terminal Assistant (GTA) - Multi-Agent Architecture

GTA é um assistente inteligente de terminal que utiliza uma arquitetura multi-agente para executar operações Git, criar e editar arquivos de código, e gerar testes unitários automaticamente.

## Arquitetura

O sistema utiliza agentes especializados, cada um focado em um domínio específico:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git Agent     │    │   Code Agent     │    │   Test Agent    │
│                 │    │                  │    │                 │
│ • Commit msgs   │    │ • File creation  │    │ • Read code     │
│ • Git commands  │    │ • Code editing   │    │ • Generate tests│
│ • Status/diff   │    │ • Content gen    │    │ • Test analysis │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Orchestrator    │
                    │                 │
                    │ • Route requests│
                    │ • Coordinate    │
                    │ • Session mgmt  │
                    └─────────────────┘
```

## Características Principais

### 1. **GitAgent** - Especialista em Versionamento
- Gera mensagens de commit inteligentes seguindo Conventional Commits
- Executa comandos git com segurança
- Analisa diffs e status do repositório

### 2. **CodeAgent** - Especialista em Código
- Cria arquivos de código em qualquer linguagem
- Edita arquivos existentes de forma inteligente
- Gera código limpo e funcional com base em descrições

### 3. **TestAgent** - Especialista em Testes
- Gera testes unitários para arquivos de código
- Suporta múltiplos frameworks (pytest, jest, junit, etc.)
- Analisa código para sugerir casos de teste

## Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd tcc

# Execute o instalador
./install.sh

# O comando 'gta' estará disponível globalmente
```

## Uso

### Modo Interativo
```bash
gta
```

### Comando Único
```bash
gta "criar arquivo calculator.py com funções matemáticas"
gta "gerar testes para calculator.py"
gta "commit com mensagem descritiva"
```

## Exemplos de Comandos

### Operações Git
```bash
# Status do repositório
git status

# Criar commit inteligente
commit com mensagem descritiva

# Adicionar tudo e commitar
adicionar tudo e commitar
```

### Criação de Arquivos
```bash
# Criar arquivo Python
criar arquivo utils.py com função de validação de email

# Criar componente React
criar arquivo Button.jsx componente React de botão

# Criar classe Java
criar arquivo User.java classe de usuário com getters e setters
```

### Geração de Testes
```bash
# Gerar testes para um arquivo
gerar testes para utils.py

# Analisar o que precisa ser testado
analisar utils.py para testes

# Gerar testes para múltiplos arquivos
gerar testes para arquivos python
```

### Comandos de Terminal
```bash
# Comandos diretos funcionam normalmente
ls -la
pwd
cat arquivo.txt
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

- Python 3.8+
- Ollama com modelo qwen3:14b
- Git instalado

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
│   ├── test_agent.py      # Geração de testes
│   └── orchestrator.py    # Roteador de requisições
├── main.py                # Entry point
├── llm_backend.py         # Configuração do LLM
├── install.sh             # Instalador
├── gta                    # Script global
└── README.md              # Este arquivo
```

## Troubleshooting

### Erro: Modelo não encontrado
Certifique-se de que o Ollama está rodando e o modelo está instalado:
```bash
ollama pull qwen3:14b
```

### Comando não reconhecido
Use `help` para ver comandos disponíveis ou seja mais específico na requisição.

## Licença

MIT

## Créditos

Desenvolvido com arquitetura multi-agente para máxima eficiência e manutenibilidade. 