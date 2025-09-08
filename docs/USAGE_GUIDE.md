# Guia de Uso - Sistema Híbrido GTA

## Como Usar as Novas Funcionalidades

### 🆕 Validação Automática de Código

O sistema agora valida automaticamente todo código Python gerado, corrigindo erros em tempo real.

#### Exemplos Práticos

```bash
# Geração de código com validação automática
gta "criar arquivo calculator.py com funções matemáticas básicas"

# O sistema automaticamente:
# 1. Gera o código Python
# 2. Valida sintaxe e imports
# 3. Corrige erros se encontrados
# 4. Retorna código validado
```

**Saída Esperada:**
```
✅ Código gerado e validado com sucesso
📁 Arquivo criado: calculator.py
🔍 Validação: aprovado (0 erros, 1 sugestão)
💡 Sugestão: considere usar f-strings para formatação
```

### 🆕 Retry Inteligente

O sistema aprende com falhas anteriores e adapta sua abordagem automaticamente.

#### Como Funciona

```bash
# Primeira tentativa falha por erro de sintaxe
gta "criar classe complexa com herança múltipla"

# Sistema detecta padrão de erro e ajusta contexto:
# - Ativa correção automática mais rigorosa
# - Aumenta validação de sintaxe
# - Enriquece prompt com exemplos de boas práticas
```

**Exemplo de Retry Automático:**
```
❌ Tentativa 1: Erro de sintaxe na linha 15
🔄 Tentativa 2: Contexto aprimorado aplicado
✅ Sucesso: Código gerado e validado
```

### 🆕 Feedback de Erro Estruturado

Erros agora são categorizados com sugestões específicas de correção.

#### Categorias de Erro

**Syntax Errors:**
```
🔴 Erro de Sintaxe
📍 Linha 12: missing colon after if statement
💡 Sugestão: adicione ':' após a condição if
🔄 Retry recomendado: Sim (alta probabilidade de correção)
```

**Import Errors:**
```
🔴 Erro de Import
📍 Módulo 'requests' não encontrado
💡 Sugestão: adicione 'pip install requests' ou use módulo padrão
🔄 Retry recomendado: Sim (verificação de dependências ativada)
```

**File System Errors:**
```
🔴 Erro de Sistema de Arquivos  
📍 Diretório '/non/existent/path' não existe
💡 Sugestão: criar diretórios pai automaticamente
🔄 Retry recomendado: Sim (criação de diretórios habilitada)
```

### 🆕 Planejamento Dinâmico

O sistema replaneja automaticamente quando detecta padrões de falha persistentes.

#### Triggers de Replanning

1. **Múltiplas falhas consecutivas** (>2 retries)
2. **Baixa confiança** (<0.5) em resultados
3. **Erros não recuperáveis** (permissões, recursos)

```bash
# Exemplo de replanning automático
gta "criar aplicação web completa com banco de dados"

# Execução:
# 1. Plano inicial: 5 tarefas
# 2. Tarefa 3 falha múltiplas vezes
# 3. Sistema detecta complexidade excessiva
# 4. Replanning: quebra tarefa 3 em sub-tarefas
# 5. Execução bem-sucedida com novo plano
```

### 🆕 Saída Estruturada com Metadados

Todos os resultados agora incluem metadados detalhados para melhor observabilidade.

#### Estrutura de Resposta

```json
{
  "task_type": "code_generation",
  "status": "completed", 
  "success": true,
  "output": "Código gerado com sucesso",
  "confidence": 0.92,
  "validation_result": {
    "valid": true,
    "errors": [],
    "warnings": ["Consider using type hints"],
    "suggestions": ["Add docstrings to functions"]
  },
  "metadata": {
    "execution_time": 2.34,
    "retry_count": 0,
    "validation_performed": true,
    "correction_attempted": false,
    "agent_type": "code"
  }
}
```

## Comandos Avançados

### Configuração de Comportamento

```bash
# Ativar validação mais rigorosa
GTA_CODE_VALIDATION_STRICT=true gta "criar código Python"

# Desativar correção automática
GTA_CODE_AUTO_CORRECTION=false gta "gerar função complexa"

# Ajustar limite de retry
GTA_WORKFLOW_MAX_RETRIES=5 gta "tarefa complexa"
```

### Modo Debug

```bash
# Ativar logs detalhados
GTA_DEBUG=true gta -i

# Ver decisões de retry/replan
GTA_WORKFLOW_DEBUG=true gta "tarefa multi-step"
```

## Exemplos de Workflows Complexos

### 1. Desenvolvimento com Testes

```bash
gta "criar módulo de autenticação com testes unitários completos"
```

**Fluxo Automático:**
1. Gera código do módulo
2. Valida sintaxe automaticamente
3. Corrige erros se encontrados
4. Gera testes unitários
5. Valida testes
6. Executa testes para verificação
7. Reporta cobertura e qualidade

### 2. Projeto Multi-Arquivo

```bash
gta "criar API REST completa com FastAPI, incluindo modelos, rotas e testes"
```

**Fluxo Inteligente:**
1. Planeja estrutura do projeto
2. Executa tarefas em ordem otimizada
3. Valida cada arquivo gerado
4. Detecta dependências entre arquivos
5. Ajusta imports automaticamente
6. Retry inteligente em falhas
7. Replanning se necessário

### 3. Análise e Refatoração

```bash
gta "analisar código legado em src/ e sugerir refatorações"
```

**Capacidades Avançadas:**
1. Análise estática de qualidade
2. Detecção de code smells
3. Sugestões de melhoria estruturadas
4. Validação de refatorações propostas
5. Testes de regressão automáticos

## Integração com IDEs

### VS Code

```json
// settings.json
{
  "terminal.integrated.env.linux": {
    "GTA_CODE_VALIDATION_ENABLED": "true",
    "GTA_WORKFLOW_DEBUG": "false"
  }
}
```

### PyCharm

```bash
# External Tool Configuration
Program: /path/to/gta
Arguments: "$FilePrompt$"
Working Directory: $ProjectFileDir$
```

## Troubleshooting das Novas Funcionalidades

### Validação Muito Restritiva

```bash
# Reduzir rigor da validação
export GTA_CODE_VALIDATION_STRICT=false

# Desativar correção automática temporariamente
export GTA_CODE_AUTO_CORRECTION=false
```

### Retry Excessivo

```bash
# Limitar número de retries
export GTA_WORKFLOW_MAX_RETRIES=2

# Aumentar threshold de confiança
export GTA_WORKFLOW_CONFIDENCE_THRESHOLD=0.7
```

### Performance

```bash
# Desativar validação para arquivos grandes
export GTA_CODE_VALIDATION_SIZE_LIMIT=10000

# Cache de validações
export GTA_VALIDATION_CACHE_ENABLED=true
```

## Monitoramento e Métricas

### Logs de Sistema

```bash
# Ver estatísticas de execução
gta stats

# Histórico de retries
gta history --retries

# Análise de padrões de erro
gta analyze-errors --last-week
```

### Dashboard de Qualidade

```bash
# Relatório de qualidade de código
gta quality-report

# Métricas de validação
gta validation-metrics
```

## Personalização Avançada

### Custom Validators

```python
# ~/.gta/custom_validators.py
def validate_security_patterns(code: str) -> Dict[str, Any]:
    """Validador customizado para padrões de segurança."""
    # Implementação personalizada
    pass
```

### Custom Retry Strategies

```python
# ~/.gta/retry_strategies.py
def custom_retry_logic(error_history: List[Dict]) -> Dict[str, Any]:
    """Lógica customizada de retry baseada em histórico."""
    # Implementação personalizada
    pass
```

## Melhores Práticas

### 1. **Estruture Requisições Claramente**
```bash
# ✅ Bom
gta "criar classe User com métodos de validação de email e senha"

# ❌ Vago  
gta "fazer algo com usuários"
```

### 2. **Use Contexto Quando Necessário**
```bash
# Para projetos existentes, forneça contexto
gta "adicionar método de autenticação à classe existente User em models.py"
```

### 3. **Aproveite a Validação Automática**
```bash
# Deixe o sistema validar e corrigir automaticamente
# Não desative validação a menos que necessário
```

### 4. **Monitore Métricas de Qualidade**
```bash
# Revise regularmente relatórios de qualidade
gta quality-report --project

# Acompanhe tendências de erro
gta error-trends --last-month
```

### 5. **Configure Ambiente Apropriadamente**
```bash
# Para desenvolvimento: validação rigorosa
export GTA_CODE_VALIDATION_STRICT=true

# Para prototipagem: mais permissivo  
export GTA_CODE_VALIDATION_STRICT=false
export GTA_WORKFLOW_MAX_RETRIES=1
```
