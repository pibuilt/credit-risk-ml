from pydantic import BaseModel, Field
from typing import List


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    term: str
    emp_length: str
    home_ownership: str

    annual_inc: float = Field(..., gt=0)
    verification_status: str
    purpose: str

    dti: float
    delinq_2yrs: int

    fico_range_low: int
    fico_range_high: int

    open_acc: int
    pub_rec: int

    revol_bal: float
    revol_util: float

    total_acc: int

    initial_list_status: str
    application_type: str

    mort_acc: int
    pub_rec_bankruptcies: int

    emp_title: str
    title: str


class PredictionRequest(BaseModel):
    data: List[LoanApplication]