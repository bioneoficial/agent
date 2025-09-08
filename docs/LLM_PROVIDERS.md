# Documentação do GTA - Índice

## 📚 Documentação Completa

### 📖 Documentos Principais
- [`README.md`](../README.md) - Visão geral e instalação do sistema
- [`HYBRID_WORKFLOW_SYSTEM.md`](./HYBRID_WORKFLOW_SYSTEM.md) - Documentação técnica do sistema híbrido 🆕
- [`USAGE_GUIDE.md`](./USAGE_GUIDE.md) - Guia de uso das novas funcionalidades 🆕
- [`LLM_PROVIDERS.md`](./LLM_PROVIDERS.md) - Configuração de provedores LLM

### 🏗️ Arquitetura do Sistema
O GTA utiliza uma arquitetura multi-agente híbrida com:
- **Sistema de Workflow Inteligente** - Execução com retry e replanning automático
- **Validação Automática de Código** - Correção em tempo real usando LLM
- **Saída Estruturada** - Modelos Pydantic para consistência de dados
- **Análise de Confiança** - Decisões baseadas em scores de confiança

### 🆕 Funcionalidades Avançadas
- Validação automática de sintaxe Python
- Correção automática de erros via LLM
- Sistema de retry inteligente com aprendizado
- Feedback estruturado de erros
- Replanning dinâmico baseado em contexto
- Metadados detalhados de execução

---

# Guia de Provedores LLM para o GTA

Este documento detalha como configurar e utilizar diferentes provedores LLM com o Git Terminal Assistant (GTA). agora suporta múltiplos provedores de LLM, permitindo escolher entre modelos locais (Ollama) ou APIs comerciais (OpenAI, Anthropic, Google, Cohere, Azure).

## Provedores Suportados

- **Ollama** (padrão) - Modelos locais, privacidade total
- **OpenAI** - GPT-4, GPT-3.5-turbo
- **Anthropic** - Claude 3 Opus, Sonnet
- **Google** - Gemini Pro, PaLM
- **Cohere** - Command, Command-Light
- **Azure OpenAI** - Deployment empresarial do OpenAI

## Configuração Rápida

### 1. Copie o arquivo de exemplo

```bash
cp .env.example .env
```

### 2. Configure seu provedor preferido

Edite `.env` e defina:

```bash
# Escolha o provedor
LLM_PROVIDER=openai  # ou anthropic, google, cohere, azure_openai, ollama

# Configure o modelo
LLM_MODEL=gpt-4

# Adicione sua API key
OPENAI_API_KEY=sua-chave-aqui
```

### 3. Instale as dependências

Opção B (recomendado) — arquivos separados:

```bash
# Núcleo (core)
pip install -r requirements.txt

# Provedores online
pip install -r requirements-providers.txt

# Tudo (núcleo + provedores)
pip install -r requirements-all.txt
```

Alternativa — instalar por provedor (apenas se/quando for usar):

```bash
# Para OpenAI
pip install langchain-openai

# Para Anthropic
pip install langchain-anthropic

# Para Google
pip install langchain-google-genai

# Para Cohere
pip install langchain-cohere

# Para Azure OpenAI
pip install langchain-openai
```

## Exemplos de Configuração

### OpenAI

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

### Anthropic Claude

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini

```bash
LLM_PROVIDER=google
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=...
```

### Ollama Local

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

### Azure OpenAI

```bash
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

## Configuração por Agente

Você pode usar modelos diferentes para cada agente:

```bash
# Modelo padrão
LLM_MODEL=llama3.1:8b

# Modelos específicos por agente
GIT_AGENT_MODEL=gpt-4        # Commits mais inteligentes
CODE_AGENT_MODEL=claude-3     # Melhor geração de código
CHAT_AGENT_MODEL=llama3.1:8b  # Chat local e privado
```

## Configuração Avançada

### Parâmetros de Modelo

```bash
LLM_TEMPERATURE=0.3    # Criatividade (0.0-1.0)
LLM_MAX_TOKENS=2000    # Tamanho máximo da resposta
LLM_TIMEOUT=60         # Timeout em segundos
```

### Fallback Automático

Se o provedor principal falhar, o sistema tentará alternativas:

```bash
FALLBACK_PROVIDERS=ollama,openai  # Tenta Ollama, depois OpenAI
FALLBACK_MODEL=llama3.1:8b        # Modelo para fallback
```

## Privacidade e Custos

| Provedor | Privacidade | Custo | Performance |
|----------|-------------|-------|-------------|
| Ollama | ⭐⭐⭐⭐⭐ Total | Grátis | Depende do hardware |
| OpenAI | ⭐⭐ API externa | Pago por uso | Excelente |
| Anthropic | ⭐⭐ API externa | Pago por uso | Excelente |
| Google | ⭐⭐ API externa | Pago/Grátis limitado | Muito boa |
| Cohere | ⭐⭐ API externa | Pago/Grátis limitado | Boa |
| Azure | ⭐⭐⭐ Empresarial | Pago por uso | Excelente |

## Solução de Problemas

### Provider não encontrado

```
Error: OpenAI provider requires: pip install langchain-openai
```

**Solução:** Instale a dependência necessária.

### API Key inválida

```
Error: OpenAI API key not found. Set OPENAI_API_KEY in .env
```

**Solução:** Verifique se o arquivo `.env` existe e contém a chave correta.

### Fallback para Ollama

Se houver problemas com o provedor configurado, o sistema automaticamente tentará usar Ollama como fallback. Certifique-se de ter o Ollama instalado e rodando:

```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Baixar modelo padrão
ollama pull llama3.1:8b

# Iniciar servidor
ollama serve
```

## Testando a Configuração

Após configurar, teste com:

```bash
# Teste interativo
gta -i
> What LLM provider am I using?

# Teste de commit (requer alterações staged)
gta 'commit'
```

O sistema mostrará qual provedor e modelo está sendo usado durante a inicialização.

## Mantendo Compatibilidade

O sistema mantém total compatibilidade com a configuração anterior via `config.py`. Se não houver arquivo `.env`, o comportamento padrão com Ollama será mantido.
