from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, NeutrosophicSupervisor, SupervisorAgent
from agent_squad.agents.supervisor_agent import SupervisorAgentOptions
from agent_squad.storage import InMemoryChatStorage
from agent_squad.types import ConversationMessage, ParticipantRole


class MockBedrockLLMAgent(BedrockLLMAgent):
    def __init__(self, options, response_text="Mock response"):
        super().__init__(options)
        self.response_text = response_text

    async def process_request(self, *args, **kwargs):
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": self.response_text}]
        )


@pytest.fixture
def mock_boto3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client


def mock_storage():
    storage = MagicMock(spec=InMemoryChatStorage)
    storage.save_chat_messages = AsyncMock()
    storage.fetch_chat = AsyncMock(return_value=[])
    storage.fetch_all_chats = AsyncMock(return_value=[])
    return storage


def make_agent(name, response_text):
    return MockBedrockLLMAgent(
        BedrockLLMAgentOptions(name=name, description=f"{name} description"),
        response_text=response_text,
    )


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_adds_consensus_to_agent_responses(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    team = [
        make_agent("Billing Agent", "The answer is clear and complete for billing invoices."),
        make_agent("Tech Agent", "Maybe this is unclear and could be a conflict."),
    ]
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=team,
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Billing Agent", "content": "Help"},
        {"recipient": "Tech Agent", "content": "Help"},
    ])

    assert "<neutrosophic_consensus>" in response
    assert "action:" in response
    assert "T:" in response
    assert "I:" in response
    assert "F:" in response
    assert "Billing Agent:" in response
    assert "Tech Agent:" in response


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_preserves_no_match_message(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=[],
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([])

    assert response == "No agent matches for the request:[]"


@pytest.mark.asyncio
async def test_existing_supervisor_agent_still_concatenates_responses(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    team = [make_agent("Team Member", "Plain response")]
    supervisor = SupervisorAgent(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=team,
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Team Member", "content": "Help"},
    ])

    assert response == "Team Member: Plain response"
    assert "<neutrosophic_consensus>" not in response
