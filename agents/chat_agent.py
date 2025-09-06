from typing import Dict, Any, Optional
import re
from agents.base_agent import BaseAgent

class ChatAgent(BaseAgent):
    """Agente especializado em responder perguntas e fornecer informações sem executar ações"""
    
    def __init__(self):
        # System prompt para o agente de chat
        system_prompt = """Você é um assistente especializado em responder perguntas sobre programação, desenvolvimento e operações.
Seu papel é fornecer informações úteis, explicações e orientações sem executar ações.
Quando o usuário fizer uma pergunta, forneça uma resposta clara e informativa.
Responda sempre em português, com explicações claras e exemplos quando relevante.
Nunca assuma que uma pergunta é um comando para criar ou modificar arquivos."""
        
        super().__init__("ChatAgent", system_prompt)
    
    def can_handle(self, request: str) -> bool:
        """Verifica se este agente pode lidar com a solicitação"""
        req = (request or "").strip().lower()
        if not req:
            return False
        
        # Qualquer entrada terminando com '?' é tratada como pergunta/informação
        if req.endswith('?'):
            return True
        
        # Detectores de pergunta (PT e EN)
        question_starters = [
            # PT
            r'^(como(\s+eu)?\s+)', r'^(o que(\s+é|são)?\s+)', r'^(qual(\s+é|são)?\s+)',
            r'^(quais(\s+são)?\s+)', r'^(por que\s+)', r'^(quando\s+)', r'^(onde\s+)',
            r'^(me\s+(explique|diga|informe|ajude)\s+)', r'^(explique\s+)',
            # EN
            r'^(what\s+)', r'^(who\s+)', r'^(why\s+)', r'^(when\s+)', r'^(where\s+)', r'^(how\s+)',
            r'^(tell me\s+)', r'^(show me\s+)', r'^(list|describe|explain)\s+'
        ]
        for pattern in question_starters:
            if re.search(pattern, req):
                return True
        
        # Heurísticas adicionais
        if any(phrase in req for phrase in [
            'o que é', 'o que sao', 'o que são', 'como funciona',
            'what is', 'how to', 'how do i', 'explain', 'describe']):
            return True
        
        return False
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa a solicitação de informação e retorna uma resposta usando o LLM"""
        # Extrair informações de contexto relevantes para enriquecer o prompt
        request_lower = request.lower()
        
        # Extract conversation history from memory
        memory = context.get("memory") if context else None
        history_text = ""
        if memory:
            # Get last 5 exchanges for context
            history_snippets = []
            msgs = memory.chat_memory.messages  # list of messages
            # Combine each pair of user/assistant into "User: ..., AI: ..."
            for i in range(0, len(msgs), 2):
                if i+1 < len(msgs):
                    user_msg = msgs[i].content
                    ai_msg = msgs[i+1].content
                    history_snippets.append(f"Usuário: {user_msg}\nAI: {ai_msg}")
            
            if history_snippets:
                history_text = "\n\n".join(history_snippets[-5:]) + "\n\n"
        
        # Detectar possível contexto na pergunta
        context_info = self._extract_context_from_request(request)
        
        # Construir prompt para o LLM com histórico de conversação e contexto
        prompt = f"""{history_text}O usuário fez a seguinte pergunta:
"{request}"

Forneça uma resposta informativa e útil, sem assumir que é um comando para executar ações.
Responda em português, com explicações claras e exemplos quando relevante.
Use o histórico da conversa acima para fornecer respostas mais contextualizadas e coerentes.
"""
        
        # Adicionar contexto ao prompt se houver informações relevantes
        if context_info:
            prompt += "\n\nInformações de contexto detectadas:\n"
            
            if context_info.get('language'):
                prompt += f"- A pergunta parece estar relacionada à linguagem {context_info['language'].upper()}\n"
                
            if context_info.get('framework'):
                prompt += f"- A pergunta menciona o framework {context_info['framework'].upper()}\n"
                
            if context_info.get('filename'):
                prompt += f"- A pergunta menciona o arquivo '{context_info['filename']}'\n"
                
            if context_info.get('topic'):
                prompt += f"- O tópico da pergunta parece ser sobre {context_info['topic']}\n"
                
            if 'test' in context_info.get('topic', ''):
                prompt += "\nIncluir informações sobre como executar testes, comandos relevantes e opções úteis para o contexto específico.\n"
        
        # Gerar resposta com o LLM
        try:
            response = self.invoke_llm(prompt)
            
            return {
                "success": True,
                "output": response,
                "type": "information",
                "agent": "ChatAgent"
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao processar a pergunta: {str(e)}",
                "type": "information",
                "error": str(e)
            }
    
    def _extract_context_from_request(self, request: str) -> Dict[str, str]:
        """Extrai informações de contexto da solicitação para enriquecer o prompt"""
        context = {}
        request_lower = request.lower()
        
        # Detectar linguagem mencionada
        language_patterns = {
            'python': [r'\bpy\b', r'\bpython\b', r'\bpytest\b', r'\bunittest\b', r'\.py\b'],
            'javascript': [r'\bjs\b', r'\bjavascript\b', r'\bnode\b', r'\bnpm\b', r'\bjest\b', r'\bmocha\b', r'\.js\b'],
            'typescript': [r'\bts\b', r'\btypescript\b', r'\.ts\b'],
            'java': [r'\bjava\b', r'\bjunit\b', r'\bmaven\b', r'\.java\b'],
            'c#': [r'\bc#\b', r'\bdotnet\b', r'\bnunit\b', r'\bxunit\b', r'\.cs\b'],
            'ruby': [r'\bruby\b', r'\brspec\b', r'\.rb\b'],
            'go': [r'\bgo\b', r'\bgolang\b', r'\.go\b']
        }
        
        for lang, patterns in language_patterns.items():
            if any(re.search(pattern, request_lower) for pattern in patterns):
                context['language'] = lang
                break
        
        # Detectar framework de teste mencionado
        test_frameworks = {
            'pytest': [r'\bpytest\b'],
            'unittest': [r'\bunittest\b'],
            'jest': [r'\bjest\b'],
            'mocha': [r'\bmocha\b'],
            'junit': [r'\bjunit\b'],
            'testng': [r'\btestng\b'],
            'nunit': [r'\bnunit\b'],
            'xunit': [r'\bxunit\b'],
            'rspec': [r'\brspec\b']
        }
        
        for framework, patterns in test_frameworks.items():
            if any(re.search(pattern, request_lower) for pattern in patterns):
                context['framework'] = framework
                break
        
        # Detectar nome de arquivo
        file_match = re.search(r'(?:arquivo|código|teste[s]?)\s+(?:de|para|do|no)?\s+([a-zA-Z0-9_\./]+\.[a-zA-Z0-9]+)', request_lower)
        if file_match:
            context['filename'] = file_match.group(1)
            
            # Inferir linguagem da extensão se não detectada anteriormente
            if 'language' not in context:
                ext = context['filename'].split('.')[-1].lower()
                lang_by_ext = {
                    'py': 'python', 
                    'js': 'javascript',
                    'ts': 'typescript',
                    'java': 'java',
                    'cs': 'c#',
                    'rb': 'ruby',
                    'go': 'go'
                }
                if ext in lang_by_ext:
                    context['language'] = lang_by_ext[ext]
        
        # Detectar tópico principal
        topics = {
            'test': [r'\bteste[s]?\b', r'\btestar\b', r'\brodar\s+teste[s]?\b', r'\bexecutar\s+teste[s]?\b', r'\btdd\b'],
            'git': [r'\bgit\b', r'\bcommit\b', r'\bbranch\b', r'\bmerge\b', r'\bpull\b', r'\bpush\b'],
            'file': [r'\barquivo[s]?\b', r'\bcriar\b', r'\beditar\b', r'\bmanipular\b'],
            'debug': [r'\bdebug\b', r'\bdebugar\b', r'\berro\b', r'\bexceção\b', r'\bdebugging\b'],
            'performance': [r'\bperformance\b', r'\botimizar\b', r'\bvelocidade\b', r'\blento\b'],
            'deploy': [r'\bdeploy\b', r'\bimplantar\b', r'\bpublicação\b', r'\bprodução\b']
        }
        
        for topic, patterns in topics.items():
            if any(re.search(pattern, request_lower) for pattern in patterns):
                context['topic'] = topic
                break
        
        return context
