//---------------------------------------------------------------------------

#include <vcl.h>
#pragma hdrstop
#include "DisplayGUI.h"
#include "AreaDialog.h"
#include "TriangulatPoly.h"

//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TAreaConfirm *AreaConfirm;
//---------------------------------------------------------------------------
__fastcall TAreaConfirm::TAreaConfirm(TComponent* Owner)
	: TForm(Owner)
{
 ColorCount=0;
}
//---------------------------------------------------------------------------
void __fastcall TAreaConfirm::OkButtonClick(TObject *Sender)
{
 DWORD Row,Count,i;

 Count=Form1->Areas->Count;
 for (i = 0; i < Count; i++)
 {
  TArea *Area = (TArea *)Form1->Areas->Items[i];
  if (Area->Name==AreaName->Text) {
   ShowMessage("Duplicate Name (Invalid)");
   return;
   }
 }

 triangulatePoly(Form1->AreaTemp->Points,Form1->AreaTemp->NumPoints,
				 &Form1->AreaTemp->Triangles);

 Form1->AreaTemp->Name=AreaName->Text;
 Form1->AreaTemp->Color=ColorBox1->Selected;
 Form1->Areas->Add(Form1->AreaTemp);



 Form1->AreaListView->Items->BeginUpdate();
 Form1->AreaListView->Items->Add();
 Row=Form1->AreaListView->Items->Count-1;
 Form1->AreaListView->Items->Item[Row]->Caption=Form1->AreaTemp->Name;
 Form1->AreaListView->Items->Item[Row]->Data=Form1->AreaTemp;
 Form1->AreaListView->Items->Item[Row]->SubItems->Add("0");    // Count column
 Form1->AreaListView->Items->Item[Row]->SubItems->Add("");     // Color column
 Form1->AreaListView->Items->EndUpdate();


 Form1->AreaTemp=NULL;
 Form1->Insert->Enabled=true;
 Form1->Complete->Enabled=false;
 Form1->Cancel->Enabled=false;
 Form1->LoadARTCCBoundaries1->Enabled=true;
// if (Form1->Areas->Count>0)  Form1->Delete->Enabled=true;
 //else   Form1->Delete->Enabled=false;
 ColorCount++;
 Close();
}
//---------------------------------------------------------------------------
void __fastcall TAreaConfirm::CancelButtonClick(TObject *Sender)
{

 TArea *Temp;
 Temp= Form1->AreaTemp;
 Form1->AreaTemp=NULL;
 delete  Temp;
 Form1->Insert->Enabled=true;
 Form1->Complete->Enabled=false;
 Form1->Cancel->Enabled=false;
 Form1->LoadARTCCBoundaries1->Enabled=true;
 // if (Form1->Areas->Count>0)  Form1->Delete->Enabled=true;
 //else   Form1->Delete->Enabled=false;

 Close();
}
//---------------------------------------------------------------------------
void __fastcall TAreaConfirm::ShowDialog(void)
{
  AreaName->Text="Area"+IntToStr((int)ColorCount+1);
  ShowModal();
}
//---------------------------------------------------------------------------

