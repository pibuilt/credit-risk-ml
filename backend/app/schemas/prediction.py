from pydantic import BaseModel, Field
from typing import List



from typing import Optional

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    dti: float
    fico_range_low: int
    # All other fields are now optional
    term: Optional[str] = None
    emp_length: Optional[str] = None
    home_ownership: Optional[str] = None
    verification_status: Optional[str] = None
    purpose: Optional[str] = None
    delinq_2yrs: Optional[int] = None
    fico_range_high: Optional[int] = None
    open_acc: Optional[int] = None
    pub_rec: Optional[int] = None
    revol_bal: Optional[float] = None
    revol_util: Optional[float] = None
    total_acc: Optional[int] = None
    initial_list_status: Optional[str] = None
    application_type: Optional[str] = None
    mort_acc: Optional[int] = None
    pub_rec_bankruptcies: Optional[int] = None
    emp_title: Optional[str] = None
    title: Optional[str] = None


class PredictionRequest(BaseModel):
    data: List[LoanApplication]