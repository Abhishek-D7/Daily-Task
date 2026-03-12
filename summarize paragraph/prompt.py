def get_analysis_prompt(text: str) -> list[dict]:
    return [
        {
            "role": "system", 
            "content": (
                "You are a strict data extraction system. Your output must be ONLY a valid, parseable JSON object. "
                "Do not include markdown formatting (like ```json), backticks, or conversational filler. "
                "You are bound by a strict rule: You may ONLY use information explicitly stated in the provided text."
            )
        },
        {
            "role": "user", 
            "content": (
                "Extract the following information from the text below. \n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. If the information for a field is present, extract it concisely.\n"
                "2. If the information is NOT explicitly mentioned in the text, you MUST output exactly \"Not in context\" for string values, or [\"Not in context\"] for lists.\n"
                "3. Do not infer, guess, or use outside knowledge.\n\n"
                "REQUIRED JSON SCHEMA:\n"
                "{\n"
                '  "summary": "String. A 1-2 sentence summary of the text.",\n'
                '  "action_items": ["List of strings. Specific tasks assigned to people or teams."],\n'
                '  "risks": ["List of strings. Potential problems, delays, or threats mentioned."],\n'
                '  "priority_tasks": ["List of strings. Tasks explicitly noted as urgent, immediate, or top priority."]\n'
                "}\n\n"
                f"TEXT TO ANALYZE:\n{text}"
            )
        }
    ]
