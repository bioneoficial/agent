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



### Comandos de Terminal

O GTA suporta uma variedade de comandos de terminal nativos, incluindo:

- Navegação e listagem: `ls`, `pwd`, `cd`, `mkdir`, `find`
- Gerenciamento de arquivos: `cat`, `grep`, `head`, `tail`
- Processos e sistema: `ps`, `top`, `htop`
- Rede: `ping`, `ssh`, `curl`, `wget`

Exemplos:
```bash
# Comandos diretos funcionam normalmente
ls -la
cd meudiretorio
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
```
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