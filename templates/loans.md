# Loans (LendingClub) - Serialization Template

## Fields

| Field | Type | Description |
|-------|------|-------------|
| emp_title | text | Employment title |
| emp_length | numeric | Years at current employer |
| state | categorical | US state |
| homeownership | categorical | OWN/RENT/MORTGAGE |
| annual_income | numeric | Annual income |
| verified_income | categorical | Not Verified / Verified / Source Verified |
| debt_to_income | numeric | Debt-to-income ratio |
| delinq_2y | numeric | Delinquencies in past 2 years |
| total_credit_lines | numeric | Total credit lines |
| total_credit_limit | numeric | Total credit limit |
| account_never_delinq_percent | numeric | % of time account never delinquent |
| loan_purpose | categorical | Loan purpose |
| loan_amount | numeric | Requested loan amount |
| term | numeric | Loan term in months |
| interest_rate | numeric | Interest rate |
| grade | categorical | Loan grade (A-G) |

## Template

```
The applicant works as {emp_title} for {emp_length} years in {state},
with {homeownership} housing and an annual income of {annual_income}.
Their verified income status is {verified_income} with a debt-to-income ratio of {debt_to_income}.
They have {delinq_2y} delinquencies in the past 2 years, {total_credit_lines} total credit lines,
and a total credit limit of {total_credit_limit}.
Their account never-delinquency percentage is {account_never_delinq_percent}.
They applied for a {loan_purpose} loan of {loan_amount} with a term of {term} months
at {interest_rate}% interest, graded as {grade}.
```

## Example Output

> The applicant works as mechanic for 0.7 years in TX,
> with MORTGAGE housing and an annual income of 0.03.
> Their verified income status is Verified with a debt-to-income ratio of 0.04.
> They have 0.25 delinquencies in the past 2 years, 0.5 total credit lines,
> and a total credit limit of 0.15.
> Their account never-delinquency percentage is 0.95.
> They applied for a medical loan of 0.17 with a term of 0.0 months
> at 0.08% interest, graded as A.

## Target Mapping

- `Fully Paid` → 1 (good)
- `Charged Off` / `Default` → 0 (default)
- `Current` and other intermediate statuses → dropped
