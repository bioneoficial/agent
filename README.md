# macOS AI Terminal Assistant (TCC Project)

Este projeto é um assistente de IA para o terminal macOS, desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) em Sistemas de Informação. O objetivo é criar um agente capaz de auxiliar em tarefas diárias no terminal, como automação de comandos Git, navegação no sistema de arquivos, e execução de comandos comuns.

## Tecnologias Utilizadas

*   **LLM:** Ollama (rodando localmente com modelos como `phi3:mini`)
*   **Framework do Agente:** LangChain (Python)
*   **Linguagem:** Python 3
*   **Ambiente:** macOS

## Configuração do Ambiente

1.  **Pré-requisitos:**
    *   macOS
    *   [Homebrew](https://brew.sh/) instalado.
    *   Python 3.10+ instalado (pode ser via Homebrew: `brew install python`)
    *   Git instalado (pode ser via Homebrew: `brew install git`)

2.  **Clone o Repositório (se aplicável):**
    ```bash
    # Se você for clonar de um repositório remoto:
    # git clone git@github.com:bioneoficial/agent.git
    # cd agent
    ```

3.  **Instale o Ollama:**
    ```bash
    brew install ollama
    ```

4.  **Inicie o Ollama e Baixe um Modelo:**
    Abra um terminal separado e execute:
    ```bash
    ollama serve
    ```
    Em outro terminal (ou após o `serve` iniciar), baixe um modelo (ex: `phi3:mini`):
    ```bash
    ollama pull phi3:mini
    ```
    *Nota: Mantenha o terminal com `ollama serve` rodando enquanto utiliza o agente.*

5.  **Crie e Ative o Ambiente Virtual Python:**
    No diretório raiz do projeto:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

6.  **Instale as Dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

## Como Rodar o Agente

1.  Certifique-se de que o serviço `ollama serve` está rodando em um terminal separado.
2.  No diretório raiz do projeto, com o ambiente virtual (`venv`) ativado, execute:
    ```bash
    python agent.py
    ```
3.  Você verá um prompt como `(venv) macOS-AI>`. Digite seus comandos ou perguntas para o agente.
    *   Exemplos:
        *   `Liste os arquivos na pasta atual.`
        *   `Qual o status do git?`
        *   `Crie uma nova branch chamada feature/nova-funcionalidade.`

## Estrutura do Projeto (Inicial)

*   `agent.py`: Script principal que contém a lógica do agente LangChain.
*   `requirements.txt`: Lista das dependências Python.
*   `.gitignore`: Especifica arquivos e diretórios a serem ignorados pelo Git.
*   `venv/`: Diretório do ambiente virtual Python (ignorado pelo Git).
*   `README.md`: Este arquivo.

## Próximos Passos e Funcionalidades Planejadas

*   Melhorar a interpretação de comandos complexos.
*   Adicionar mais ferramentas específicas para Git (commits, merges, etc.).
*   Integrar com outras ferramentas de linha de comando.
*   Permitir configuração de um alias no terminal para acesso rápido. 