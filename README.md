---

title: Dataops Env
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false

---

# `dataops-env`

`dataops-env` is an OpenEnv benchmark for training and evaluating agents on
multi-step data operations work. Instead of a single obvious cleanup action, an
agent must inspect messy business tables, choose corrective actions in the right
order, preserve valid-but-unusual records, and know when the table is truly
ready for validation.

It exposes the standard `reset()`, `step(action)`, and `state()` interface,
ships with a production-ready FastAPI server and Docker image, and includes a
reproducible OpenAI-compatible baseline runner.

## Benchmark Purpose

Many toy data-cleaning tasks reward shallow pattern matching. Real operational
data work is harder:

- duplicates may be safe to remove, but conflicting rows require judgment
- some malformed values should be normalized, while unusual valid values must be preserved
- deletion is often the riskiest action, not the default fix
- agents need partial credit for progress, but strong penalties for repeated mistakes

`dataops-env` is designed to capture those decisions in a compact benchmark that
is still easy to run, validate, and deploy in the OpenEnv ecosystem.

## Why It Feels Real

The environment models common enterprise data quality problems:

- exact duplicates in customer or vendor master data
- missing required fields
- inconsistent casing in names and locations
- invalid email and phone formats
- conflicting records for the same real-world entity
- uniqueness constraints such as shared-email violations
- trap rows that look suspicious but are actually valid

Agents are rewarded for minimal corrective behavior and punished for destructive
or repetitive actions. That makes the environment useful for both learning and
evaluation.

## Task Families

The benchmark keeps the hackathon-friendly `easy`, `medium`, and `hard` task
structure, while each family now contains deterministic variants so policies
cannot overfit a single table.

1. `easy`
   Remove duplicates and fill missing required fields.
2. `medium`
   Remove duplicates, normalize casing, and repair invalid emails.
3. `hard`
   Resolve conflicts, enforce unique-email constraints, fix invalid formats,
   and preserve valid trap rows.

Each task definition includes:

- `goal`
- `difficulty`
- `variant_id`
- `required_columns`
- `hidden_issues`
- `constraints`
- `expected_outcome`
- `max_steps`

## Learning Signals

The environment provides both dense rewards and a deterministic final score:

- partial rewards for duplicate removal, normalization, and filling missing values
- step costs and no-progress penalties to discourage random actions
- escalating penalties for repeated mistakes
- destructive-action penalties for harmful deletions
- proactive hints after recurring failures
- final task scoring on a strict `0.0` to `1.0` scale

The final task score and the visible validation failures are produced from the
same explicit rule set, reducing mismatch between what the agent sees and how it
is ultimately judged.

## Action Space

Agents interact with the environment through a typed `Action` object.

Supported action types:

- `remove_duplicate`
  Remove one row from an exact duplicate group. Can be called with an explicit
  `row_id`, or the environment can choose the default duplicate target.
- `fill_missing`
  Fill a missing field on a target row. Requires `column` and `value`, and may
  also include `row_id`.
- `normalize_column`
  Apply deterministic normalization to a supported column such as `name`,
  `city`, `email`, or `phone`.
- `delete_row`
  Delete a row when doing so resolves a structural issue like a conflict or a
  uniqueness violation. Requires `row_id`.
- `validate`
  Signal that the agent believes the table is ready for completion.
- `noop`
  Explicitly take no action. This is allowed but penalized when unresolved
  issues remain.

Typed action schema:

- `action_id: Optional[str]`
- `action_type: Literal["remove_duplicate", "fill_missing", "normalize_column", "delete_row", "validate", "noop"]`
- `column: Optional[str]`
- `row_id: Optional[int]`
- `value: Optional[str]`

Validation rules:

- `delete_row` requires `row_id`
- `normalize_column` requires `column`
- `fill_missing` requires `column` and `value`

Example actions:

```json
{"action_id":"step-001","action_type":"remove_duplicate","row_id":33}
{"action_id":"step-002","action_type":"fill_missing","row_id":35,"column":"email","value":"peak.systems@example.com"}
{"action_id":"step-003","action_type":"normalize_column","column":"email"}
{"action_id":"step-004","action_type":"validate"}
```

## Observation Space

The environment returns a typed `Observation` object after `reset()` and each
call to `step()`.

Observation fields:

- `goal: str`
  Natural-language description of what the agent should accomplish.
- `table: List[Dict[str, Any]]`
  Current JSON-serializable table snapshot.
- `issues: List[str]`
  Human-readable unresolved issues and validation failures.
- `history: List[str]`
  Ordered record of previous actions/events in the current episode.
- `mistakes: Dict[str, int]`
  Counts of repeated mistake categories tracked during the episode.
- `hints: List[str]`
  Proactive or reactive guidance derived from issue state and prior failures.
- `progress: float`
  Normalized progress estimate in `[0.0, 1.0]`.
- `steps_remaining: int`
  Number of remaining actions before the episode terminates.

Example observation shape:

```json
{
  "goal": "Normalize the dataset by fixing casing, removing duplicates, and correcting invalid email formats.",
  "table": [
    {"row_id": 10, "customer_id": "C100", "name": "jane miller", "city": "new york", "email": "jane.miller@example.com"}
  ],
  "issues": [
    "Rows 11 and 13 are duplicates and only one should remain."
  ],
  "history": [],
  "mistakes": {},
  "hints": [],
  "progress": 0.0,
  "steps_remaining": 9
}
```

## Expected Agent Behavior

A strong agent should behave roughly like this:

1. inspect the visible table and unresolved issues
2. remove safe duplicates first
3. repair missing or malformed values without over-editing valid rows
4. resolve structural conflicts carefully, especially in hard tasks
5. validate only when the remaining issue list is empty

Example successful baseline trace:

```text
[START] task=medium env=dataops-env model=your-model
[STEP] step=1 action=remove_duplicate(row_id=13) reward=0.37 done=false error=null
[STEP] step=2 action=normalize_column(column='email') reward=0.27 done=false error=null
[STEP] step=3 action=normalize_column(column='name') reward=0.24 done=false error=null
[STEP] step=4 action=normalize_column(column='city') reward=0.44 done=true error=null
[END] success=true steps=4 rewards=0.37,0.27,0.24,0.44
```

## Project Layout

- `env.py`: core `DataOpsEnv` implementation
- `task.py`: task families and deterministic variants
- `models.py`: typed `Action`, `Observation`, and `Reward` contracts
- `grader.py`: dense rewards, explicit validation checks, and final task scoring
- `server/app.py`: FastAPI runtime API
- `inference.py`: hybrid heuristic/model baseline runner
- `openenv.yaml`: OpenEnv metadata and task registration
- `pyproject.toml`: package metadata and server script entry point
- `Dockerfile`: production container image

## Local Setup

```bash
pip install -r requirements.txt
openenv validate
```

Run the FastAPI server:

```bash
python -m server.app
```

By default, the local server runs on port `8000`.

Or use the packaged entry point:

```bash
server
```

## API

Health check:

```bash
curl http://localhost:8000/health
```

Create a session with an optional seed and task selection:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 0, "task_name": "easy"}'
```

Step the environment:

```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{"action_id":"step-001","action_type":"validate"}'
```

Read internal state:

```bash
curl "http://localhost:8000/state"
```

## Baseline Inference

The baseline runner now combines deterministic local planning with optional
model arbitration. The local planner proposes ranked candidate actions from the
visible table state, and the model is constrained to choose only from those
candidates. This avoids many common failure modes such as invalid actions,
repeated no-op loops, and reckless deletion choices.

Run it with an OpenAI-compatible endpoint:

```bash
set HF_TOKEN=your_token
set MODEL_NAME=your_model
set API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

Key properties:

- strict `[START]`, `[STEP]`, and `[END]` output formatting
- fixed task ordering for reproducibility
- retry logic for invalid or blocked model suggestions
- strong heuristic fallback when the model is unavailable
- action filtering based on prior no-progress or errorful behavior

## Docker

Build:

```bash
docker build -t dataops-env .
```

Run locally:

```bash
docker run -p 8000:8000 dataops-env
```

## Hugging Face Spaces Notes

For Hugging Face `Docker` Spaces, the container should normally listen on port
`7860`, or the Space must be explicitly configured to expect a different
internal port.

If you keep the current container on port `8000`, make sure your Space is
configured with:

```yaml
app_port: 8000
```

If you want the simplest Hugging Face Spaces setup, change the container to use
port `7860` instead:

```dockerfile
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Then local Docker testing would become:

```bash
docker run -p 7860:7860 dataops-env
curl http://localhost:7860/health
```

## Submission Notes

- `openenv validate` passes
- the server and Docker image run successfully
- the packaged benchmark supports multi-mode deployment
- the default baseline now completes the public task families deterministically

Leaderboard performance will still depend on the quality of the external model,
but the repository is now structured and documented like a serious benchmark
submission rather than a starter scaffold.
