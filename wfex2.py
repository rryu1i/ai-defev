from datetime import datetime
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY não encontrada no ambiente.")

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
modelo = "meta-llama/llama-4-scout-17b-16e-instruct"


class TipoSolicitacaoCalendario(BaseModel):
    tipo_solicitacao: Literal["novo_evento", "modificar_evento", "outro"] = Field(
        description="Tipo de solicitação de calendário sendo feita"
    )
    pontuacao_confianca: float = Field(description="Pontuação de confiança entre 0 e 1")
    descricao: str = Field(description="Descrição limpa da solicitação")


class DetalhesNovoEvento(BaseModel):
    nome: str = Field(description="Nome do evento")
    data: str = Field(description="Data e hora do evento (ISO 8601)")
    duracao_minutos: int = Field(description="Duração em minutos")
    participantes: list[str] = Field(description="Lista de participantes")


class Mudanca(BaseModel):
    campo: str = Field(description="Campo a ser alterado")
    novo_valor: str = Field(description="Novo valor para o campo")


class DetalhesModificarEvento(BaseModel):
    identificador_evento: str = Field(
        description="Descrição para identificar o evento existente"
    )
    mudancas: list[Mudanca] = Field(description="Lista de mudanças a fazer")
    participantes_adicionar: list[str] = Field(
        description="Novos participantes para adicionar"
    )
    participantes_remover: list[str] = Field(description="Participantes para remover")


class RespostaCalendario(BaseModel):
    sucesso: bool = Field(description="Se a operação foi bem-sucedida")
    mensagem: str = Field(description="Mensagem de resposta amigável ao usuário")
    link_calendario: Optional[str] = Field(
        description="Link do calendário se aplicável"
    )


def rotear_solicitacao_calendario(entrada_usuario: str) -> TipoSolicitacaoCalendario:
    response = client.responses.parse(
        model=modelo,
        input="Determine se esta é uma solicitação para criar um novo evento de calendário ou modificar um existente.",
        instructions=f"Analise esta solicitação: '{entrada_usuario}'",
        text_format=TipoSolicitacaoCalendario,
    )
    return response.output_parsed


def processar_novo_evento(descricao: str) -> RespostaCalendario:
    hoje = datetime.now()
    contexto_data = f"Hoje é {hoje.strftime('%A, %d de %B de %Y')}."
    response = client.responses.parse(
        model=modelo,
        input=f"{contexto_data} Extraia detalhes para criar um novo evento de calendário.",
        instructions=f"Extraia informações estruturadas desta descrição: '{descricao}'",
        text_format=DetalhesNovoEvento,
    )
    detalhes = response.output_parsed

    return RespostaCalendario(
        sucesso=True,
        mensagem=f"Criado novo evento '{detalhes.nome}' para {detalhes.data} com {', '.join(detalhes.participantes)}",
        link_calendario=f"calendar://new?event={detalhes.nome}",
    )


def processar_modificar_evento(descricao: str) -> RespostaCalendario:
    hoje = datetime.now()
    contexto_data = f"Hoje é {hoje.strftime('%A, %d de %B de %Y')}."
    response = client.responses.parse(
        model=modelo,
        input=f"{contexto_data} Extraia detalhes para modificar um evento de calendário existente.",
        instructions=f"Extraia informações de modificação desta descrição: '{descricao}'",
        text_format=DetalhesModificarEvento,
    )
    detalhes = response.output_parsed

    mudancas_texto = ", ".join(
        [f"{m.campo} para {m.novo_valor}" for m in detalhes.mudancas]
    )

    return RespostaCalendario(
        sucesso=True,
        mensagem=f"Modificado evento '{detalhes.identificador_evento}': {mudancas_texto}",
        link_calendario=f"calendar://modify?event={detalhes.identificador_evento}",
    )


def processar_solicitacao_calendario(
    entrada_usuario: str,
) -> Optional[RespostaCalendario]:
    resultado_roteamento = rotear_solicitacao_calendario(entrada_usuario)

    if resultado_roteamento.pontuacao_confianca < 0.7:
        return None

    if resultado_roteamento.tipo_solicitacao == "novo_evento":
        return processar_novo_evento(resultado_roteamento.descricao)
    elif resultado_roteamento.tipo_solicitacao == "modificar_evento":
        return processar_modificar_evento(resultado_roteamento.descricao)
    else:
        return None


entrada_novo_evento = "Vamos agendar uma reunião de equipe na próxima terça-feira às 14h com Daniel e Alberto"
resultado = processar_solicitacao_calendario(entrada_novo_evento)
if resultado:
    print(f"Resposta: {resultado.mensagem}")
else:
    print("Solicitação não reconhecida como operação de calendário")


entrada_modificar_evento = (
    "Você pode mover a reunião de equipe com Daniel e Alberto para quarta-feira às 15h?"
)
resultado = processar_solicitacao_calendario(entrada_modificar_evento)
if resultado:
    print(f"Resposta: {resultado.mensagem}")
else:
    print("Solicitação não reconhecida como operação de calendário")


entrada_invalida = "Como está o clima hoje?"
resultado = processar_solicitacao_calendario(entrada_invalida)
if not resultado:
    print("Solicitação não reconhecida como operação de calendário")
