# German Credit Risk - Serialization Template

## Fields

| Field | Type | Description |
|-------|------|-------------|
| Age | numeric | Applicant age |
| Sex | categorical | Gender |
| Job | numeric | Job level (0-3) |
| Housing | categorical | Housing type (own/rent/free) |
| Saving accounts | categorical | Savings level (little/moderate/rich/Unknown) |
| Checking account | categorical | Checking level (little/moderate/rich/Unknown) |
| Credit amount | numeric | Requested credit amount |
| Duration | numeric | Loan duration in months |
| Purpose | categorical | Loan purpose |

## Template

```
The applicant is {Sex}, aged {Age}, with job level {Job}.
They live in {Housing} housing.
Their saving accounts are {Saving accounts} and checking account is {Checking account}.
They requested a credit amount of {Credit amount} over {Duration} months for {Purpose}.
```

## Example Output

> The applicant is male, aged 0.16, with job level 0.67.
> They live in own housing.
> Their saving accounts are Unknown and checking account is moderate.
> They requested a credit amount of 0.09 over 0.21 months for business.

## Target Mapping

- `good` → 1 (creditworthy)
- `bad` → 0 (default risk)
