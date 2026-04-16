# CLS Annotation Guide

Label each sentence with exactly one of these 7 categories.

## Labels

### `statement`
Declarative sentence containing factual or descriptive content.

**Examples:**
- "Caroline works at the hospital downtown."
- "The server runs on port 9090."
- "I went to an LGBTQ support group yesterday."
- "The adoption process requires a home study first."
- "Oil paints work better than watercolors for landscapes."

### `question`
Genuine question seeking information.  No embedded factual assertions.

**Examples:**
- "Where is the nearest hospital?"
- "What time does the meeting start?"
- "How do I reset my password?"
- "Can you explain this error?"
- "Who is responsible for the deployment?"

### `question_fact`
Question that embeds a factual assertion or presupposition.
The question form wraps actual information.

**Examples:**
- "Did you know Caroline moved to Paris?"
- "Have you heard that the company is being acquired?"
- "Is it true that the server was down for 3 hours?"
- "Do you remember when she started at the clinic?"
- "Isn't the deadline next Friday?"

### `command`
Imperative sentence — a request, instruction, or order.

**Examples:**
- "Show me the deployment logs."
- "Please restart the server."
- "Run the test suite before merging."
- "Delete the old configuration files."
- "Find all users who signed up last week."

### `greeting`
Social opener or closer.  No informational content.

**Examples:**
- "Hey!"
- "Hi there, how are you?"
- "Good morning!"
- "Hello everyone."
- "Goodbye, see you tomorrow."

### `filler`
Reaction, backchannel, or emotional response.
Contains no extractable information.

**Examples:**
- "Wow!"
- "lol"
- "That's amazing!"
- "haha"
- "No way!"
- "Sounds great!"
- "Thanks so much!"
- "You're welcome."

### `acknowledgment`
Agreement, confirmation, or understanding signal.
Short, no new information.

**Examples:**
- "OK"
- "Sure, I'll do that."
- "Got it."
- "Understood."
- "Right, makes sense."
- "Yep."
- "Agreed."

## Edge Cases

| Sentence | Label | Reasoning |
|----------|-------|-----------|
| "Yeah I think she went to Paris." | `statement` | Contains factual content despite the "yeah" opener |
| "Can you show me the logs?" | `command` | Request disguised as question — primary intent is action |
| "Oh really, she moved?" | `question_fact` | Surprise reaction + embedded fact + question |
| "I mean, it's just a prototype." | `statement` | Hedging filler ("I mean") but the content is factual |
| "Sure, the meeting is at 3." | `statement` | "Sure" is acknowledgment but the sentence adds facts |
| "Thanks for letting me know." | `filler` | Politeness, no information content |
| "OK so the plan is to migrate next week." | `statement` | "OK so" is a discourse marker, content is factual |

## Rules

1. **Primary intent wins.** If a sentence mixes categories, choose the one
   that best describes the primary communicative purpose.
2. **Facts trump reactions.** "Wow, she got the job!" → `statement`
   (the fact that she got the job is the information content).
3. **Short = likely not statement.** Sentences under 4 words are rarely
   `statement` unless very dense ("Server is down.").
4. **Questions ending in `?` that are really commands** → `command`
   ("Could you fix this?" is a polite command, not a question).
5. **When in doubt, choose `statement`.** It's the safest default for
   information extraction — false positives are better than false negatives.
