from agent_squad.agents.supervisor_agent import SupervisorAgent, SupervisorAgentOptions
from agent_squad.neutrosophic import decide, neutrosophic_consensus, score_text_response


class NeutrosophicSupervisor(SupervisorAgent):
    """Supervisor agent that adds neutrosophic consensus to team responses."""

    async def send_messages(self, messages: list[dict[str, str]]) -> str:
        """Process messages and return agent responses with neutrosophic consensus."""
        raw_response = await super().send_messages(messages)
        if not raw_response or raw_response.startswith("No agent matches"):
            return raw_response

        scored_responses = [
            (response, score_text_response(response))
            for response in self._split_agent_responses(raw_response)
        ]
        consensus = neutrosophic_consensus(score for _, score in scored_responses)
        action = decide(consensus)

        return self._format_neutrosophic_response(scored_responses, consensus, action.value)

    def _split_agent_responses(self, raw_response: str) -> list[str]:
        marker_positions: list[int] = []
        for agent in self.team:
            marker = f"{agent.name}: "
            start = raw_response.find(marker)
            while start != -1:
                marker_positions.append(start)
                start = raw_response.find(marker, start + len(marker))

        marker_positions = sorted(set(marker_positions))
        responses = []
        for index, start in enumerate(marker_positions):
            end = marker_positions[index + 1] if index + 1 < len(marker_positions) else len(raw_response)
            responses.append(raw_response[start:end].strip())

        return responses or [raw_response]

    @staticmethod
    def _format_neutrosophic_response(scored_responses, consensus, action: str) -> str:
        score_lines = [
            f"- {response}\n"
            f"  T={score.T:.2f}, I={score.I:.2f}, F={score.F:.2f}"
            for response, score in scored_responses
        ]

        return (
            "<neutrosophic_consensus>\n"
            f"action: {action}\n"
            f"T: {consensus.T:.2f}\n"
            f"I: {consensus.I:.2f}\n"
            f"F: {consensus.F:.2f}\n"
            "</neutrosophic_consensus>\n"
            "<agent_responses>\n"
            + "\n".join(score_lines)
            + "\n</agent_responses>"
        )
