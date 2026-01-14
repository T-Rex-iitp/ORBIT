//---------------------------------------------------------------------------

#include <vcl.h>
#include <new>
#include <math.h>
#include <dir.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <fileapi.h>
#include <windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#pragma comment(lib, "winmm.lib")

#pragma hdrstop

#include "DisplayGUI.h"
#include "AreaDialog.h"
#include "ntds2d.h"
#include "LatLonConv.h"
#include "PointInPolygon.h"
#include "DecodeRawADS_B.h"
#include "ght_hash_table.h"
#include "dms.h"
#include "Aircraft.h"
#include "TimeFunctions.h"
#include "SBS_Message.h"
#include "CPA.h"
#include "AircraftDB.h"
#include "csv.h"

#define AIRCRAFT_DATABASE_URL   "https://opensky-network.org/datasets/metadata/aircraftDatabase.zip"
#define AIRCRAFT_DATABASE_FILE   "aircraftDatabase.csv"
#define ARTCC_BOUNDARY_FILE      "Ground_Level_ARTCC_Boundary_Data_2025-05-15.csv"
//"https://vrs-standing-data.adsb.lol/routes.csv.gz"
#define API_SERVICE_URL_JSON  "https://vrs-standing-data.adsb.lol/routes/%.2s/%s.json"
#define API_SERVICE_URL_TXT  "https://vrs-standing-data.adsb.lol/routes/%.2s/%s.txt"
#define MAP_CENTER_LAT  40.73612;
#define MAP_CENTER_LON -80.33158;

#define BIG_QUERY_UPLOAD_COUNT 50000
#define BIG_QUERY_RUN_FILENAME  "SimpleCSVtoBigQuery.py"
// #define BIG_QUERY_CONDA_ACTIVATE_ENV "BIGQUERY_CONDA_ACTIVATE"
// #define BIG_QUERY_CONDA_ENV_NAME_ENV "BIGQUERY_CONDA_ENV"
// #define BIG_QUERY_CONDA_ACTIVATE_DEFAULT "C:\\Users\\admin\\miniconda3\\Scripts\\activate.bat"
// #define BIG_QUERY_CONDA_ENV_DEFAULT "iitp"
#define   LEFT_MOUSE_DOWN   1
#define   RIGHT_MOUSE_DOWN  2
#define   MIDDLE_MOUSE_DOWN 4


#define BG_INTENSITY   0.37
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma link "OpenGLPanel"
#pragma link "Map\libgefetch\Win64\Release\libgefetch.a"
#pragma link "Map\zlib\Win64\Release\zlib.a"
#pragma link "Map\jpeg\Win64\Release\jpeg.a"
#pragma link "Map\png\Win64\Release\png.a"
#pragma link "HashTable\Lib\Win64\Release\HashTableLib.a"
#pragma link "cspin"
#pragma link "SpeechLib_OCX"
#pragma resource "*.dfm"
TForm1 *Form1;
 //---------------------------------------------------------------------------
 static void RunPythonScript(AnsiString scriptPath,AnsiString args);
 static bool DeleteFilesWithExtension(AnsiString dirPath, AnsiString extension);
 static int FinshARTCCBoundary(void);
 static AnsiString RunPythonScriptWithOutput(AnsiString scriptPath, AnsiString args);
 //---------------------------------------------------------------------------

static char *stristr(const char *String, const char *Pattern);
static const char * strnistr(const char * pszSource, DWORD dwLength, const char * pszFind) ;

//---------------------------------------------------------------------------
uint32_t createRGB(uint8_t r, uint8_t g, uint8_t b)
{
  return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}
//---------------------------------------------------------------------------
uint32_t PopularColors[] = {
	  createRGB(255, 0, 0),      // Red
	  createRGB(0, 255, 0),      // Green
	  createRGB(0, 0, 255),      // Blue
	  createRGB(255, 255, 0),   // Yellow
	  createRGB(255, 165, 0),   // Orange
	  createRGB(255, 192, 203), // Pink
	  createRGB(0, 255, 255),   // Cyan
	  createRGB(255, 0, 255),  // Magenta
	  createRGB(255,255, 255),    // White
	  //createRGB(0, 0, 0),        // Black
	  createRGB(128,128,128),      // Gray
	  createRGB(165,42,42)    // Brown
  };

  int NumColors = sizeof(PopularColors) / sizeof(PopularColors[0]);
 unsigned int CurrentColor=0;


 //---------------------------------------------------------------------------
 typedef struct
{
   union{ 
     struct{ 
	 System::Byte Red;
	 System::Byte Green;
	 System::Byte Blue;
	 System::Byte Alpha;
     }; 
     struct{ 
     TColor Cl; 
     }; 
     struct{ 
     COLORREF Rgb; 
     }; 
   };

}TMultiColor;


//---------------------------------------------------------------------------
static const char * strnistr(const char * pszSource, DWORD dwLength, const char * pszFind)
{
	DWORD        dwIndex   = 0;
	DWORD        dwStrLen  = 0;
	const char * pszSubStr = NULL;

	// check for valid arguments
	if (!pszSource || !pszFind)
	{
		return pszSubStr;
	}

	dwStrLen = strlen(pszFind);

	// can pszSource possibly contain pszFind?
	if (dwStrLen > dwLength)
	{
		return pszSubStr;
	}

	while (dwIndex <= dwLength - dwStrLen)
	{
		if (0 == strnicmp(pszSource + dwIndex, pszFind, dwStrLen))
		{
			pszSubStr = pszSource + dwIndex;
			break;
		}

		dwIndex ++;
	}

	return pszSubStr;
}
//---------------------------------------------------------------------------
static char *stristr(const char *String, const char *Pattern)
{
  char *pptr, *sptr, *start;
  size_t  slen, plen;

  for (start = (char *)String,pptr  = (char *)Pattern,slen  = strlen(String),plen  = strlen(Pattern);
       slen >= plen;start++, slen--)
      {
            /* find start of pattern in string */
            while (toupper(*start) != toupper(*Pattern))
            {
                  start++;
                  slen--;

                  /* if pattern longer than string */

                  if (slen < plen)
                        return(NULL);
            }

            sptr = start;
            pptr = (char *)Pattern;

            while (toupper(*sptr) == toupper(*pptr))
            {
                  sptr++;
                  pptr++;

                  /* if end of pattern then pattern was found */

                  if ('\0' == *pptr)
                        return (start);
            }
      }
   return(NULL);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
__fastcall TForm1::TForm1(TComponent* Owner)
	: TForm(Owner)
{
  AircraftDBPathFileName=ExtractFilePath(ExtractFileDir(Application->ExeName)) +AnsiString("..\\AircraftDB\\")+AIRCRAFT_DATABASE_FILE;
  ARTCCBoundaryDataPathFileName=ExtractFilePath(ExtractFileDir(Application->ExeName)) +AnsiString("..\\ARTCC_Boundary_Data\\")+ARTCC_BOUNDARY_FILE;
  BigQueryPath=ExtractFilePath(ExtractFileDir(Application->ExeName)) +AnsiString("..\\BigQuery\\");
  BigQueryPythonScript= BigQueryPath+ AnsiString(BIG_QUERY_RUN_FILENAME);
  DeleteFilesWithExtension(BigQueryPath, "csv");
  BigQueryLogFileName=BigQueryPath+"BigQuery.log";
  DeleteFileA(BigQueryLogFileName.c_str());
  CurrentSpriteImage=0;
  InitDecodeRawADS_B();
  
  // Initialize speech transcription paths
  AnsiString HomeDir = ExtractFilePath(ExtractFileDir(Application->ExeName));
  // Use conda iitp environment Python directly
  SpeechPythonPath = "C:\\Users\\admin\\miniconda3\\envs\\iitp\\python.exe"; // Use iitp conda environment
  // Fix path: go up from Win64/Release to ADS-B-Display, then up to AI-Enabled-IFTA, then to speech
  SpeechTranscribeScriptPath = HomeDir + AnsiString("..\\..\\speech\\transcribe.py");
  SpeechTranscribeThread = NULL;
  IsRecordingVoice = false;
  RecordedAudioPath = "";
  AudioRecorder = new TAudioRecorder();
  
  // Initialize airport tracking
  TrackingAirport = false;
  AirportName = "";
  AirportLat = 0.0;
  AirportLon = 0.0;
  AirportRadiusMiles = 100.0;
  AirportListItem = NULL;
  
  RecordRawStream=NULL;
  PlayBackRawStream=NULL;
  TrackHook.Valid_CC=false;
  TrackHook.Valid_CPA=false;

  HashTable = ght_create(50000);

  if ( !HashTable)
	{
	  throw Sysutils::Exception("Create Hash Failed");
	}
  ght_set_rehash(HashTable, TRUE);

  AreaTemp=NULL;
  Areas= new TList;

 MouseDown=false;

 MapCenterLat=MAP_CENTER_LAT;
 MapCenterLon=MAP_CENTER_LON;

 LoadMapFromInternet=false;
 MapComboBox->ItemIndex=GoogleMaps;
 //MapComboBox->ItemIndex=SkyVector_VFR;
 //MapComboBox->ItemIndex=SkyVector_IFR_Low;
 //MapComboBox->ItemIndex=SkyVector_IFR_High;
 LoadMap(MapComboBox->ItemIndex);

 g_EarthView->m_Eye.h /= pow(1.3,18);//pow(1.3,43);
 SetMapCenter(g_EarthView->m_Eye.x, g_EarthView->m_Eye.y);
 TimeToGoTrackBar->Position=120;
 BigQueryCSV=NULL;
 BigQueryRowCount=0;
 BigQueryFileCount=0;
 InitAircraftDB(AircraftDBPathFileName);
 Form1->SpVoice1->Rate=2; // Set Rate of Voice
 Form1->SpVoice1->Volume=100;  //Set Volume of Voice
 printf("init complete\n");
}
//---------------------------------------------------------------------------
__fastcall TForm1::~TForm1()
{
 Timer1->Enabled=false;
 Timer2->Enabled=false;
 if (AudioRecorder) 
 {
	 AudioRecorder->StopRecording();
	 delete AudioRecorder;
 }
 if (SpeechTranscribeThread) 
 {
	 if (!SpeechTranscribeThread->Finished)
	 {
		 SpeechTranscribeThread->Terminate();
		 SpeechTranscribeThread->WaitFor();
	 }
	 delete SpeechTranscribeThread;
 }
 delete g_EarthView;
 if (g_GETileManager) delete g_GETileManager;
 delete g_MasterLayer;
 delete g_Storage;
 if (LoadMapFromInternet)
 {
   if (g_Keyhole) delete g_Keyhole;
 }

}
//---------------------------------------------------------------------------
void __fastcall  TForm1::SetMapCenter(double &x, double &y)
{
  double siny;
  x=(MapCenterLon+0.0)/360.0;
  siny=sin((MapCenterLat * M_PI) / 180.0);
  siny = fmin(fmax(siny, -0.9999), 0.9999);
  y=(log((1 + siny) / (1 - siny)) / (4 * M_PI));
}
//---------------------------------------------------------------------------
void __fastcall TForm1::ObjectDisplayInit(TObject *Sender)
{
	glViewport(0,0,(GLsizei)ObjectDisplay->Width,(GLsizei)ObjectDisplay->Height);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glEnable (GL_LINE_STIPPLE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    NumSpriteImages=MakeAirplaneImages();
	MakeAirTrackFriend();
	MakeAirTrackHostile();
	MakeAirTrackUnknown();
	MakePoint();
	MakeTrackHook();
	g_EarthView->Resize(ObjectDisplay->Width,ObjectDisplay->Height);
	glPushAttrib (GL_LINE_BIT);
	glPopAttrib ();
    printf("OpenGL Version %s\n",glGetString(GL_VERSION));
}
//---------------------------------------------------------------------------

void __fastcall TForm1::ObjectDisplayResize(TObject *Sender)
{
	 double Value;
	//ObjectDisplay->Width=ObjectDisplay->Height;
	glViewport(0,0,(GLsizei)ObjectDisplay->Width,(GLsizei)ObjectDisplay->Height);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glEnable (GL_LINE_STIPPLE);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	g_EarthView->Resize(ObjectDisplay->Width,ObjectDisplay->Height);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::ObjectDisplayPaint(TObject *Sender)
{

 if (DrawMap->Checked)glClearColor(0.0,0.0,0.0,0.0);
 else	glClearColor(BG_INTENSITY,BG_INTENSITY,BG_INTENSITY,0.0);

 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 g_EarthView->Animate();
 g_EarthView->Render(DrawMap->Checked);
 g_GETileManager->Cleanup();
 Mw1 = Map_w[1].x-Map_w[0].x;
 Mw2 = Map_v[1].x-Map_v[0].x;
 Mh1 = Map_w[1].y-Map_w[0].y;
 Mh2 = Map_v[3].y-Map_v[0].y;

 xf=Mw1/Mw2;
 yf=Mh1/Mh2;

 DrawObjects();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Timer1Timer(TObject *Sender)
{
 __int64 CurrentTime;

 CurrentTime=GetCurrentTimeInMsec();
 SystemTime->Caption=TimeToChar(CurrentTime);

 ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::DrawObjects(void)
{
  double ScrX, ScrY;
  int    ViewableAircraft=0;

  glEnable( GL_LINE_SMOOTH );
  glEnable( GL_POINT_SMOOTH );
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glLineWidth(3.0);
  glPointSize(4.0);
  glColor4f(1.0, 1.0, 1.0, 1.0);

  LatLon2XY(MapCenterLat,MapCenterLon, ScrX, ScrY);

  glBegin(GL_LINE_STRIP);
  glVertex2f(ScrX-20.0,ScrY);
  glVertex2f(ScrX+20.0,ScrY);
  glEnd();

  glBegin(GL_LINE_STRIP);
  glVertex2f(ScrX,ScrY-20.0);
  glVertex2f(ScrX,ScrY+20.0);
  glEnd();


  uint32_t *Key;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data,*DataCPA;

  DWORD i,j,Count;

  if (AreaTemp)
  {
   glPointSize(3.0);
	for (DWORD i = 0; i <AreaTemp->NumPoints ; i++)
		LatLon2XY(AreaTemp->Points[i][1],AreaTemp->Points[i][0],
				  AreaTemp->PointsAdj[i][0],AreaTemp->PointsAdj[i][1]);

   glBegin(GL_POINTS);
   for (DWORD i = 0; i <AreaTemp->NumPoints ; i++)
	{
	glVertex2f(AreaTemp->PointsAdj[i][0],
			   AreaTemp->PointsAdj[i][1]);
	}
	glEnd();
   glBegin(GL_LINE_STRIP);
   for (DWORD i = 0; i <AreaTemp->NumPoints ; i++)
	{
	glVertex2f(AreaTemp->PointsAdj[i][0],
			   AreaTemp->PointsAdj[i][1]);
	}
	glEnd();
  }
	Count=Areas->Count;
	for (i = 0; i < Count; i++)
	 {
	   TArea *Area = (TArea *)Areas->Items[i];
	   TMultiColor MC;

	   MC.Rgb=ColorToRGB(Area->Color);
	   if (Area->Selected)
	   {
		  glLineWidth(4.0);
		  glPushAttrib (GL_LINE_BIT);
		  glLineStipple (3, 0xAAAA);
	   }


	   glColor4f(MC.Red/255.0, MC.Green/255.0, MC.Blue/255.0, 1.0);
	   glBegin(GL_LINE_LOOP);
	   for (j = 0; j <Area->NumPoints; j++)
	   {
		LatLon2XY(Area->Points[j][1],Area->Points[j][0], ScrX, ScrY);
		glVertex2f(ScrX,ScrY);
	   }
	  glEnd();
	   if (Area->Selected)
	   {
		glPopAttrib ();
		glLineWidth(2.0);
	   }

	   glColor4f(MC.Red/255.0, MC.Green/255.0, MC.Blue/255.0, 0.4);

	   for (j = 0; j <Area->NumPoints; j++)
	   {
		LatLon2XY(Area->Points[j][1],Area->Points[j][0],
				  Area->PointsAdj[j][0],Area->PointsAdj[j][1]);
	   }
	  TTriangles *Tri=Area->Triangles;

	  while(Tri)
	  {
		glBegin(GL_TRIANGLES);
		glVertex2f(Area->PointsAdj[Tri->indexList[0]][0],
				   Area->PointsAdj[Tri->indexList[0]][1]);
		glVertex2f(Area->PointsAdj[Tri->indexList[1]][0],
				   Area->PointsAdj[Tri->indexList[1]][1]);
		glVertex2f(Area->PointsAdj[Tri->indexList[2]][0],
				   Area->PointsAdj[Tri->indexList[2]][1]);
		glEnd();
		Tri=Tri->next;
	  }
	 }

    AircraftCountLabel->Caption=IntToStr((int)ght_size(HashTable));
	for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
			  Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
	{
	  if (Data->HaveLatLon)
	  {
		ViewableAircraft++;
	   glColor4f(1.0, 1.0, 1.0, 1.0);

	   LatLon2XY(Data->Latitude,Data->Longitude, ScrX, ScrY);
	   //DrawPoint(ScrX,ScrY);
	   if (Data->HaveSpeedAndHeading)   glColor4f(1.0, 0.0, 1.0, 1.0);
	   else
		{
		 Data->Heading=0.0;
		 glColor4f(1.0, 0.0, 0.0, 1.0);
		}

	   DrawAirplaneImage(ScrX,ScrY,1.5,Data->Heading,Data->SpriteImage);
	   glRasterPos2i(ScrX+30,ScrY-10);
	   ObjectDisplay->Draw2DText(Data->HexAddr);

	   if ((Data->HaveSpeedAndHeading) && (TimeToGoCheckBox->State==cbChecked))
	   {
		double lat,lon,az;
		if (VDirect(Data->Latitude,Data->Longitude,
					Data->Heading,Data->Speed/3060.0*TimeToGoTrackBar->Position ,&lat,&lon,&az)==OKNOERROR)
		  {
			 double ScrX2, ScrY2;
			 LatLon2XY(lat,lon, ScrX2, ScrY2);
             glColor4f(1.0, 1.0, 0.0, 1.0);
			 glBegin(GL_LINE_STRIP);
			 glVertex2f(ScrX,ScrY);
			 glVertex2f(ScrX2,ScrY2);
			 glEnd();
		  }
	   }
	 }
	}
 ViewableAircraftCountLabel->Caption=ViewableAircraft;
 if (TrackHook.Valid_CC)
 {

		Data= (TADS_B_Aircraft *)ght_get(HashTable, sizeof(TrackHook.ICAO_CC), (void *)&TrackHook.ICAO_CC);
		if (Data)
		{
		ICAOLabel->Caption=Data->HexAddr;
		if (Data->HaveFlightNum)
		  {
           FlightNumLabel->Caption=Data->FlightNum;
           if (Data->RequestedRoute==false)
           {
             _di_IHTTPResponse Repsonse;
             char _GetStr[1024];
             Data->RequestedRoute=true;
		     snprintf (_GetStr, sizeof(_GetStr), API_SERVICE_URL_TXT, Data->FlightNum, Data->FlightNum);
             //printf("%s\n", _GetStr);
             Repsonse=NetHTTPClientRoute->Get(AnsiString(_GetStr));
             if (Repsonse->StatusCode==200)
             {
              AnsiString Route=Repsonse->ContentAsString(TEncoding::ASCII);
              if (strlen(Route.c_str())<sizeof(Data->Route))
              {
               strcpy(Data->Route,Route.c_str());
               Data->HaveRoute=true;
              }
             }
            // else  printf("No Route Error\n");
           }
           if (Data->HaveRoute)
           {
              RouteLabel->Caption=Data->Route;
           }
           else RouteLabel->Caption="UNKNOWN";

		  }
		else FlightNumLabel->Caption="N/A";
        if (Data->HaveLatLon)
		{
		 CLatLabel->Caption=DMS::DegreesMinutesSecondsLat(Data->Latitude).c_str();
		 CLonLabel->Caption=DMS::DegreesMinutesSecondsLon(Data->Longitude).c_str();
        }
        else
        {
         CLatLabel->Caption="N/A";
		 CLonLabel->Caption="N/A";
        }
        if (Data->HaveSpeedAndHeading)
        {
		 SpdLabel->Caption=FloatToStrF(Data->Speed, ffFixed,12,2)+" KTS  VRATE:"+FloatToStrF(Data->VerticalRate,ffFixed,12,2);
		 HdgLabel->Caption=FloatToStrF(Data->Heading, ffFixed,12,2)+" DEG";
        }
        else
        {
 		 SpdLabel->Caption="N/A";
		 HdgLabel->Caption="N/A";
        }
        if (Data->Altitude)
		 AltLabel->Caption= FloatToStrF(Data->Altitude, ffFixed,12,2)+" FT";
		else AltLabel->Caption="N/A";

		MsgCntLabel->Caption="Raw: "+IntToStr((int)Data->NumMessagesRaw)+" SBS: "+IntToStr((int)Data->NumMessagesSBS);
		TrkLastUpdateTimeLabel->Caption=TimeToChar(Data->LastSeen);

        glColor4f(1.0, 0.0, 0.0, 1.0);
        LatLon2XY(Data->Latitude,Data->Longitude, ScrX, ScrY);
        DrawTrackHook(ScrX, ScrY);
        }

		else
        {
		 TrackHook.Valid_CC=false;
		 ICAOLabel->Caption="N/A";
		 FlightNumLabel->Caption="N/A";
         CLatLabel->Caption="N/A";
		 CLonLabel->Caption="N/A";
         SpdLabel->Caption="N/A";
		 HdgLabel->Caption="N/A";
		 AltLabel->Caption="N/A";
		 MsgCntLabel->Caption="N/A";
         TrkLastUpdateTimeLabel->Caption="N/A";
        }
 }
 if (TrackHook.Valid_CPA)
 {
  bool CpaDataIsValid=false;
  DataCPA= (TADS_B_Aircraft *)ght_get(HashTable, sizeof(TrackHook.ICAO_CPA), (void *)&TrackHook.ICAO_CPA);
  if ((DataCPA) && (TrackHook.Valid_CC))
	{

	  double tcpa,cpa_distance_nm, vertical_cpa;
	  double lat1, lon1,lat2, lon2, junk;
	  if (computeCPA(Data->Latitude, Data->Longitude, Data->Altitude,
					 Data->Speed,Data->Heading,
					 DataCPA->Latitude, DataCPA->Longitude, DataCPA->Altitude,
					 DataCPA->Speed,DataCPA->Heading,
					 tcpa,cpa_distance_nm, vertical_cpa))
	  {
		if (VDirect(Data->Latitude,Data->Longitude,
					Data->Heading,Data->Speed/3600.0*tcpa,&lat1,&lon1,&junk)==OKNOERROR)
		{
		  if (VDirect(DataCPA->Latitude,DataCPA->Longitude,
					  DataCPA->Heading,DataCPA->Speed/3600.0*tcpa,&lat2,&lon2,&junk)==OKNOERROR)
		   {
			 glColor4f(0.0, 1.0, 0.0, 1.0);
			 glBegin(GL_LINE_STRIP);
			 LatLon2XY(Data->Latitude,Data->Longitude, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 LatLon2XY(lat1,lon1, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 glEnd();
			 glBegin(GL_LINE_STRIP);
			 LatLon2XY(DataCPA->Latitude,DataCPA->Longitude, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 LatLon2XY(lat2,lon2, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 glEnd();
			 glColor4f(1.0, 0.0, 0.0, 1.0);
			 glBegin(GL_LINE_STRIP);
			 LatLon2XY(lat1,lon1, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 LatLon2XY(lat2,lon2, ScrX, ScrY);
			 glVertex2f(ScrX, ScrY);
			 glEnd();
			 CpaTimeValue->Caption=TimeToChar(tcpa*1000);
			 CpaDistanceValue->Caption= FloatToStrF(cpa_distance_nm, ffFixed,10,2)+" NM VDIST: "+IntToStr((int)vertical_cpa)+" FT";
			 CpaDataIsValid=true;
		   }
		}
	  }
	}
   if (!CpaDataIsValid)
   {
	TrackHook.Valid_CPA=false;
	CpaTimeValue->Caption="None";
	CpaDistanceValue->Caption="None";
   }
 }
}
//---------------------------------------------------------------------------
void __fastcall TForm1::ObjectDisplayMouseDown(TObject *Sender,
	  TMouseButton Button, TShiftState Shift, int X, int Y)
{

 if (Button==mbLeft)
   {
	if (Shift.Contains(ssCtrl))
	{

	}
	else
	{
	 g_MouseLeftDownX = X;
	 g_MouseLeftDownY = Y;
	 g_MouseDownMask |= LEFT_MOUSE_DOWN ;
	 g_EarthView->StartDrag(X, Y, NAV_DRAG_PAN);
	}
  }
 else if (Button==mbRight)
  {
  if (AreaTemp)
   {
	if (AreaTemp->NumPoints<MAX_AREA_POINTS)
	{
	  AddPoint(X, Y);
	}
	else ShowMessage("Max Area Points Reached");
   }
  else
   {
   if (Shift.Contains(ssCtrl))   HookTrack(X,Y,true);
   else  HookTrack(X,Y,false);
   }
  }

 else if (Button==mbMiddle)  ResetXYOffset();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::ObjectDisplayMouseUp(TObject *Sender,
	  TMouseButton Button, TShiftState Shift, int X, int Y)
{
  if (Button == mbLeft) g_MouseDownMask &= ~LEFT_MOUSE_DOWN;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::ObjectDisplayMouseMove(TObject *Sender,
	  TShiftState Shift, int X, int Y)
{
 int X1,Y1;
 double VLat,VLon;
 int i;
 Y1=(ObjectDisplay->Height-1)-Y;
 X1=X;
 if  ((X1>=Map_v[0].x) && (X1<=Map_v[1].x) &&
	  (Y1>=Map_v[0].y) && (Y1<=Map_v[3].y))

  {
   pfVec3 Point;
   VLat=atan(sinh(M_PI * (2 * (Map_w[1].y-(yf*(Map_v[3].y-Y1))))))*(180.0 / M_PI);
   VLon=(Map_w[1].x-(xf*(Map_v[1].x-X1)))*360.0;
   Lat->Caption=DMS::DegreesMinutesSecondsLat(VLat).c_str();
   Lon->Caption=DMS::DegreesMinutesSecondsLon(VLon).c_str();
   Point[0]=VLon;
   Point[1]=VLat;
   Point[2]=0.0;

   for (i = 0; i < Areas->Count; i++)
	 {
	   TArea *Area = (TArea *)Areas->Items[i];
	   if (PointInPolygon(Area->Points,Area->NumPoints,Point))
	   {
#if 0
		  MsgLog->Lines->Add("In Polygon "+ Area->Name);
#endif
       }
	 }
  }

  if (g_MouseDownMask & LEFT_MOUSE_DOWN)
  {
   g_EarthView->Drag(g_MouseLeftDownX, g_MouseLeftDownY, X,Y, NAV_DRAG_PAN);
   ObjectDisplay->Repaint();
  }

}
//---------------------------------------------------------------------------
void __fastcall TForm1::ResetXYOffset(void)
{
 SetMapCenter(g_EarthView->m_Eye.x, g_EarthView->m_Eye.y);
 ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Exit1Click(TObject *Sender)
{
 Close();
}
//---------------------------------------------------------------------------
 void __fastcall TForm1::AddPoint(int X, int Y)
 {
  double Lat,Lon;

 if (XY2LatLon2(X,Y,Lat,Lon)==0)
 {

	AreaTemp->Points[AreaTemp->NumPoints][1]=Lat;
	AreaTemp->Points[AreaTemp->NumPoints][0]=Lon;
	AreaTemp->Points[AreaTemp->NumPoints][2]=0.0;
	AreaTemp->NumPoints++;
	ObjectDisplay->Repaint();
 }
 }
 //---------------------------------------------------------------------------

wchar_t *AnsiTowchar_t(AnsiString Str)
{
wchar_t *str = new wchar_t[Str.WideCharBufSize()];
return Str.WideChar(str, Str.WideCharBufSize());
}
//---------------------------------------------------------------------------
 void __fastcall TForm1::HookTrack(int X, int Y,bool CPA_Hook)
 {
  double VLat,VLon, dlat,dlon,Range;
  int X1,Y1;
   uint32_t *Key;

   uint32_t Current_ICAO;
   double MinRange;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data;

  Y1=(ObjectDisplay->Height-1)-Y;
  X1=X;

  if  ((X1<Map_v[0].x) || (X1>Map_v[1].x) ||
	   (Y1<Map_v[0].y) || (Y1>Map_v[3].y)) return;

  VLat=atan(sinh(M_PI * (2 * (Map_w[1].y-(yf*(Map_v[3].y-Y1))))))*(180.0 / M_PI);
  VLon=(Map_w[1].x-(xf*(Map_v[1].x-X1)))*360.0;

  MinRange=16.0;

  for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
			  Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
	{
	  if (Data->HaveLatLon)
	  {
	   dlat= VLat-Data->Latitude;
	   dlon= VLon-Data->Longitude;
	   Range=sqrt(dlat*dlat+dlon*dlon);
	   if (Range<MinRange)
	   {
		Current_ICAO=Data->ICAO;
		MinRange=Range;
	   }
	  }
	}
	if (MinRange< 0.2)
	{
	  TADS_B_Aircraft * ADS_B_Aircraft =(TADS_B_Aircraft *)
			ght_get(HashTable,sizeof(Current_ICAO),
					&Current_ICAO);
	  if (ADS_B_Aircraft)
	  {
		if (!CPA_Hook)
		{
         AnsiString Text="Hooked Aircraft "+(AnsiString)GetAircraftDBInfo(ADS_B_Aircraft->ICAO) ;
         wchar_t *wtext= AnsiTowchar_t(Text);
		 TrackHook.Valid_CC=true;
		 TrackHook.ICAO_CC=ADS_B_Aircraft->ICAO;
		 printf("%s\n\n",GetAircraftDBInfo(ADS_B_Aircraft->ICAO));
         Form1->SpVoice1->Speak(wtext, SpeechVoiceSpeakFlags::SVSFlagsAsync );  // Say Text and continue
         delete wtext;
		}
		else
		{
		 TrackHook.Valid_CPA=true;
		 TrackHook.ICAO_CPA=ADS_B_Aircraft->ICAO;
        }
;
	  }

	}
	else
		{
		 if (!CPA_Hook)
		  {
		   TrackHook.Valid_CC=false;
           ICAOLabel->Caption="N/A";
		   FlightNumLabel->Caption="N/A";
		   CLatLabel->Caption="N/A";
		   CLonLabel->Caption="N/A";
		   SpdLabel->Caption="N/A";
		   HdgLabel->Caption="N/A";
		   AltLabel->Caption="N/A";
		   MsgCntLabel->Caption="N/A";
		   TrkLastUpdateTimeLabel->Caption="N/A";
		  }
		 else
		   {
			TrackHook.Valid_CPA=false;
			CpaTimeValue->Caption="None";
	        CpaDistanceValue->Caption="None";
           }
		}

 }
//---------------------------------------------------------------------------
void __fastcall TForm1::LatLon2XY(double lat,double lon, double &x, double &y)
{
 x=(Map_v[1].x-((Map_w[1].x-(lon/360.0))/xf));
 y= Map_v[3].y- (Map_w[1].y/yf)+ (asinh(tan(lat*M_PI/180.0))/(2*M_PI*yf));
}
//---------------------------------------------------------------------------
int __fastcall TForm1::XY2LatLon2(int x, int y,double &lat,double &lon )
{
  double Lat,Lon, dlat,dlon,Range;
  int X1,Y1;

  Y1=(ObjectDisplay->Height-1)-y;
  X1=x;

  if  ((X1<Map_v[0].x) || (X1>Map_v[1].x) ||
	   (Y1<Map_v[0].y) || (Y1>Map_v[3].y)) return -1;

  lat=atan(sinh(M_PI * (2 * (Map_w[1].y-(yf*(Map_v[3].y-Y1))))))*(180.0 / M_PI);
  lon=(Map_w[1].x-(xf*(Map_v[1].x-X1)))*360.0;

  return 0;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::ZoomInClick(TObject *Sender)
{
  g_EarthView->SingleMovement(NAV_ZOOM_IN);
  ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::ZoomOutClick(TObject *Sender)
{
 g_EarthView->SingleMovement(NAV_ZOOM_OUT);

 ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Purge(void)
{
  uint32_t *Key;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data;
  void *p;
  __int64 CurrentTime=GetCurrentTimeInMsec();
  __int64  StaleTimeInMs=CSpinStaleTime->Value*1000;

  if (PurgeStale->Checked==false) return;

  for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
			  Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
	{
	  if ((CurrentTime-Data->LastSeen)>=StaleTimeInMs)
	  {
	  p = ght_remove(HashTable,sizeof(*Key), Key);;
	  if (!p)
		ShowMessage("Removing the current iterated entry failed! This is a BUG\n");

	  delete Data;

	  }
	}
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Timer2Timer(TObject *Sender)
{
 Purge();
 UpdateAreaFlightCounts();  // Update flight counts in areas
}
//---------------------------------------------------------------------------
// Count flights within a polygon area (ROI)
int __fastcall TForm1::CountFlightsInArea(TArea *Area)
{
  int count = 0;
  uint32_t *Key;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data;
  pfVec3 Point;

  if (!Area || Area->NumPoints < 3) return 0;

  for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
      Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
  {
    if (Data->HaveLatLon)
    {
      // Create point for aircraft position (lon, lat, 0)
      Point[0] = Data->Longitude;
      Point[1] = Data->Latitude;
      Point[2] = 0.0;

      // Check if aircraft is inside the polygon
      if (PointInPolygon(Area->Points, Area->NumPoints, Point))
      {
        count++;
      }
    }
  }
  return count;
}
//---------------------------------------------------------------------------
// Update flight counts for all areas and display in AreaListView
void __fastcall TForm1::UpdateAreaFlightCounts(void)
{
  // Iterate through all ListView items and update counts
  for (int i = 0; i < AreaListView->Items->Count; i++)
  {
    TListItem *Item = AreaListView->Items->Item[i];
    TArea *Area = (TArea *)Item->Data;
    
    if (Area != NULL)
    {
      // This is a polygon area - count flights in polygon
      int flightCount = CountFlightsInArea(Area);
      if (Item->SubItems->Count > 0)
      {
        Item->SubItems->Strings[0] = IntToStr(flightCount);
      }
    }
    else if (TrackingAirport && Item == AirportListItem)
    {
      // This is the airport radius entry - count flights in radius
      int airportCount = CountFlightsInRadius(AirportLat, AirportLon, AirportRadiusMiles);
      if (Item->SubItems->Count > 0)
      {
        Item->SubItems->Strings[0] = IntToStr(airportCount);
      }
    }
  }
}
//---------------------------------------------------------------------------
// Calculate distance between two points in statute miles using Haversine formula
double __fastcall TForm1::CalculateDistanceMiles(double lat1, double lon1, double lat2, double lon2)
{
  const double EARTH_RADIUS_MILES = 3958.8;  // Earth radius in statute miles
  
  // Convert to radians
  double lat1Rad = lat1 * M_PI / 180.0;
  double lon1Rad = lon1 * M_PI / 180.0;
  double lat2Rad = lat2 * M_PI / 180.0;
  double lon2Rad = lon2 * M_PI / 180.0;
  
  // Haversine formula
  double dLat = lat2Rad - lat1Rad;
  double dLon = lon2Rad - lon1Rad;
  
  double a = sin(dLat / 2) * sin(dLat / 2) +
             cos(lat1Rad) * cos(lat2Rad) *
             sin(dLon / 2) * sin(dLon / 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  
  return EARTH_RADIUS_MILES * c;
}
//---------------------------------------------------------------------------
// Count flights within a radius (in miles) of a center point
int __fastcall TForm1::CountFlightsInRadius(double centerLat, double centerLon, double radiusMiles)
{
  int count = 0;
  uint32_t *Key;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data;

  for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
      Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
  {
    if (Data->HaveLatLon)
    {
      double distance = CalculateDistanceMiles(centerLat, centerLon, 
                                                Data->Latitude, Data->Longitude);
      if (distance <= radiusMiles)
      {
        count++;
      }
    }
  }
  return count;
}
//---------------------------------------------------------------------------
// JFK Button Click - Add JFK airport to tracking
void __fastcall TForm1::JFKButtonClick(TObject *Sender)
{
  // JFK Airport coordinates
  const double JFK_LAT = 40.6413;
  const double JFK_LON = -73.7781;
  const double RADIUS_MILES = 50.0;
  
  // If already tracking, remove and re-add (toggle behavior would remove, but let's just update)
  if (TrackingAirport && AirportListItem)
  {
    // Already tracking JFK, do nothing or could toggle off
    // For now, just update the count immediately
    int count = CountFlightsInRadius(JFK_LAT, JFK_LON, RADIUS_MILES);
    if (AirportListItem->SubItems->Count > 0)
    {
      AirportListItem->SubItems->Strings[0] = IntToStr(count);
    }
    return;
  }
  
  // Set up airport tracking
  TrackingAirport = true;
  AirportName = "JFK (50mi)";
  AirportLat = JFK_LAT;
  AirportLon = JFK_LON;
  AirportRadiusMiles = RADIUS_MILES;
  
  // Add entry to AreaListView
  AreaListView->Items->BeginUpdate();
  AirportListItem = AreaListView->Items->Add();
  AirportListItem->Caption = AirportName;
  AirportListItem->Data = NULL;  // No TArea data for airport radius
  
  // Initial count
  int count = CountFlightsInRadius(JFK_LAT, JFK_LON, RADIUS_MILES);
  AirportListItem->SubItems->Add(IntToStr(count));  // Count column
  AirportListItem->SubItems->Add("");               // Color column (empty for airport)
  AreaListView->Items->EndUpdate();
  
  printf("JFK Airport tracking enabled: %d flights within %.0f miles\n", count, RADIUS_MILES);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::PurgeButtonClick(TObject *Sender)
{
  uint32_t *Key;
  ght_iterator_t iterator;
  TADS_B_Aircraft* Data;
  void *p;

  for(Data = (TADS_B_Aircraft *)ght_first(HashTable, &iterator,(const void **) &Key);
			  Data; Data = (TADS_B_Aircraft *)ght_next(HashTable, &iterator, (const void **)&Key))
	{

	  p = ght_remove(HashTable,sizeof(*Key), Key);
	  if (!p)
		ShowMessage("Removing the current iterated entry failed! This is a BUG\n");

	  delete Data;

	}
}
//---------------------------------------------------------------------------
void __fastcall TForm1::InsertClick(TObject *Sender)
{
 Insert->Enabled=false;
 LoadARTCCBoundaries1->Enabled=false;
 Complete->Enabled=true;
 Cancel->Enabled=true;
 //Delete->Enabled=false;

 AreaTemp= new TArea;
 AreaTemp->NumPoints=0;
 AreaTemp->Name="";
 AreaTemp->Selected=false;
 AreaTemp->Triangles=NULL;

}
//---------------------------------------------------------------------------
void __fastcall TForm1::CancelClick(TObject *Sender)
{
 TArea *Temp;
 Temp= AreaTemp;
 AreaTemp=NULL;
 delete  Temp;
 Insert->Enabled=true;
 Complete->Enabled=false;
 Cancel->Enabled=false;
 LoadARTCCBoundaries1->Enabled=true;
 //if (Areas->Count>0)  Delete->Enabled=true;
 //else   Delete->Enabled=false;

}
//---------------------------------------------------------------------------
void __fastcall TForm1::CompleteClick(TObject *Sender)
{

  int or1=orientation2D_Polygon( AreaTemp->Points,AreaTemp->NumPoints);
  if (or1==0)
   {
	ShowMessage("Degenerate Polygon");
    CancelClick(NULL);
	return;
   }
  if (or1==CLOCKWISE)
  {
	DWORD i;

	memcpy(AreaTemp->PointsAdj,AreaTemp->Points,sizeof(AreaTemp->Points));
	for (i = 0; i <AreaTemp->NumPoints; i++)
	 {
	   memcpy(AreaTemp->Points[i],
			 AreaTemp->PointsAdj[AreaTemp->NumPoints-1-i],sizeof( pfVec3));
	 }
  }
  if (checkComplex( AreaTemp->Points,AreaTemp->NumPoints))
   {
	ShowMessage("Polygon is Complex");
	CancelClick(NULL);
	return;
   }

  AreaConfirm->ShowDialog();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::AreaListViewSelectItem(TObject *Sender, TListItem *Item,
      bool Selected)
{
   DWORD Count;
   TArea *AreaS=(TArea *)Item->Data;
   bool HaveSelected=false;
	Count=Areas->Count;
	for (unsigned int i = 0; i < Count; i++)
	 {
	   TArea *Area = (TArea *)Areas->Items[i];
	   if (Area==AreaS)
	   {
		if (Item->Selected)
		{
		 Area->Selected=true;
		 HaveSelected=true;
		}
		else
		 Area->Selected=false;
	   }
	   else
		 Area->Selected=false;

	 }
	if (HaveSelected)  Delete->Enabled=true;
	else Delete->Enabled=false;
	ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::DeleteClick(TObject *Sender)
{
 int i = 0;

 while (i < AreaListView->Items->Count)
  {
	if (AreaListView->Items->Item[i]->Selected)
	{
	 TArea *Area;
	 int Index;

	 Area=(TArea *)AreaListView->Items->Item[i]->Data;
	 for (Index = 0; Index < Areas->Count; Index++)
	 {
	  if (Area==Areas->Items[Index])
	  {
	   Areas->Delete(Index);
	   AreaListView->Items->Item[i]->Delete();
	   TTriangles *Tri=Area->Triangles;
	   while(Tri)
	   {
		TTriangles *temp=Tri;
		Tri=Tri->next;
		free(temp->indexList);
		free(temp);
	   }
	   delete Area;
	   break;
	  }
	 }
	}
	else
	{
	  ++i;
	}
  }
 //if (Areas->Count>0)  Delete->Enabled=true;
 //else   Delete->Enabled=false;

 ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::AreaListViewCustomDrawItem(TCustomListView *Sender,
	  TListItem *Item, TCustomDrawState State, bool &DefaultDraw)
{
   TRect   R;
   int Left;
  TArea *Area=(TArea *)Item->Data;

  // Clear background
  AreaListView->Canvas->Brush->Color = AreaListView->Color;
  AreaListView->Canvas->Font->Color = AreaListView->Font->Color;
  R=Item->DisplayRect(drBounds);
  AreaListView->Canvas->FillRect(R);

  // Draw Area Name (Column 0)
  AreaListView->Canvas->TextOut(2, R.Top, Item->Caption);

  Left = AreaListView->Column[0]->Width;

  // Draw Count (Column 1) - SubItems[0]
  if (Item->SubItems->Count > 0)
  {
    R=Item->DisplayRect(drBounds);
    R.Left = Left + 2;
    AreaListView->Canvas->TextOut(R.Left, R.Top, Item->SubItems->Strings[0]);
  }

  // Draw Color (Column 2) - SubItems[1]
  Left += AreaListView->Column[1]->Width;
  if (Item->SubItems->Count > 1 && Area)
  {
    R=Item->DisplayRect(drBounds);
    R.Left = Left;
    R.Right = R.Left + AreaListView->Column[2]->Width;
    AreaListView->Canvas->Brush->Color = Area->Color;
    AreaListView->Canvas->FillRect(R);
  }

  // Draw selection focus rectangle
  if (Item->Selected)
  {
    R=Item->DisplayRect(drBounds);
    AreaListView->Canvas->DrawFocusRect(R);
  }
  DefaultDraw=false;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::DeleteAllAreas(void)
{
 int i = 0;

 while (AreaListView->Items->Count)
  {

	 TArea *Area;
	 int Index;

	 Area=(TArea *)AreaListView->Items->Item[i]->Data;
	 for (Index = 0; Index < Areas->Count; Index++)
	 {
	  if (Area==Areas->Items[Index])
	  {
	   Areas->Delete(Index);
	   AreaListView->Items->Item[i]->Delete();
	   TTriangles *Tri=Area->Triangles;
	   while(Tri)
	   {
		TTriangles *temp=Tri;
		Tri=Tri->next;
		free(temp->indexList);
		free(temp);
	   }
	   delete Area;
	   break;
	  }
	 }
  }

 ObjectDisplay->Repaint();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::FormMouseWheel(TObject *Sender, TShiftState Shift,
	  int WheelDelta, TPoint &MousePos, bool &Handled)
{
 if (WheelDelta>0)
	  g_EarthView->SingleMovement(NAV_ZOOM_IN);
 else g_EarthView->SingleMovement(NAV_ZOOM_OUT);
  ObjectDisplay->Repaint();
}                                  
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void __fastcall TTCPClientRawHandleThread::HandleInput(void)
{
  modeS_message mm;
  TDecodeStatus Status;

 // Form1->MsgLog->Lines->Add(StringMsgBuffer);
  if (Form1->RecordRawStream)
  {
   __int64 CurrentTime;
   CurrentTime=GetCurrentTimeInMsec();
   Form1->RecordRawStream->WriteLine(IntToStr(CurrentTime));
   Form1->RecordRawStream->WriteLine(StringMsgBuffer);
  }

  Status=decode_RAW_message(StringMsgBuffer, &mm);
  if (Status==HaveMsg)
  {
   TADS_B_Aircraft *ADS_B_Aircraft;
   uint32_t addr;

	addr = (mm.AA[0] << 16) | (mm.AA[1] << 8) | mm.AA[2];


	ADS_B_Aircraft =(TADS_B_Aircraft *) ght_get(Form1->HashTable,sizeof(addr),&addr);
	if (ADS_B_Aircraft)
	  {
      	//Form1->MsgLog->Lines->Add("Retrived");
      }
    else
	  {
  	   ADS_B_Aircraft= new TADS_B_Aircraft;
	   ADS_B_Aircraft->ICAO=addr;
	   snprintf(ADS_B_Aircraft->HexAddr,sizeof(ADS_B_Aircraft->HexAddr),"%06X",(int)addr);
	   ADS_B_Aircraft->NumMessagesSBS=0;
       ADS_B_Aircraft->NumMessagesRaw=0;
       ADS_B_Aircraft->VerticalRate=0;
	   ADS_B_Aircraft->HaveAltitude=false;
       ADS_B_Aircraft->HaveLatLon=false;
	   ADS_B_Aircraft->HaveSpeedAndHeading=false;
	   ADS_B_Aircraft->HaveFlightNum=false;
       ADS_B_Aircraft->HaveRoute=false;
       ADS_B_Aircraft->RequestedRoute=false;
       ADS_B_Aircraft->Route[0]=NULL;
	   ADS_B_Aircraft->SpriteImage=Form1->CurrentSpriteImage;
	   if (Form1->CycleImages->Checked)
		 Form1->CurrentSpriteImage=(Form1->CurrentSpriteImage+1)%Form1->NumSpriteImages;
	   if (ght_insert(Form1->HashTable,ADS_B_Aircraft,sizeof(addr), &addr) < 0)
		  {
			printf("ght_insert Error - Should Not Happen\n");
		  }
	  }

	  RawToAircraft(&mm,ADS_B_Aircraft);
  }
  else  printf("Raw Decode Error:%d\n",Status);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::RawConnectButtonClick(TObject *Sender)
{
 IdTCPClientRaw->Host=RawIpAddress->Text;
 IdTCPClientRaw->Port=30002;

 if ((RawConnectButton->Caption=="Raw Connect") && (Sender!=NULL))
 {
  try
   {
   IdTCPClientRaw->Connect();
   TCPClientRawHandleThread = new TTCPClientRawHandleThread(true);
   TCPClientRawHandleThread->UseFileInsteadOfNetwork=false;
   TCPClientRawHandleThread->FreeOnTerminate=TRUE;
   TCPClientRawHandleThread->Resume();
   }
   catch (const EIdException& e)
   {
    ShowMessage("Error while connecting: "+e.Message);
   }
 }
 else
  {
	TCPClientRawHandleThread->Terminate();
	IdTCPClientRaw->Disconnect();
	IdTCPClientRaw->IOHandler->InputBuffer->Clear();
	RawConnectButton->Caption="Raw Connect";
	RawPlaybackButton->Enabled=true;
  }
 }
//---------------------------------------------------------------------------
void __fastcall TForm1::IdTCPClientRawConnected(TObject *Sender)
{
   //SetKeepAliveValues(const AEnabled: Boolean; const ATimeMS, AInterval: Integer);
   IdTCPClientRaw->Socket->Binding->SetKeepAliveValues(true,60*1000,15*1000);
   RawConnectButton->Caption="Raw Disconnect";
   RawPlaybackButton->Enabled=false;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::IdTCPClientRawDisconnected(TObject *Sender)
{
  TCPClientRawHandleThread->Terminate();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::RawRecordButtonClick(TObject *Sender)
{
 if (RawRecordButton->Caption=="Raw Record")
 {
  if (RecordRawSaveDialog->Execute())
   {
	// First, check if the file exists.
	if (FileExists(RecordRawSaveDialog->FileName))
	  ShowMessage("File "+RecordRawSaveDialog->FileName+"already exists. Cannot overwrite.");
	else
	{
		// Open a file for writing. Creates the file if it doesn't exist, or overwrites it if it does.
	RecordRawStream= new TStreamWriter(RecordRawSaveDialog->FileName, false);
	if (RecordRawStream==NULL)
	  {
		ShowMessage("Cannot Open File "+RecordRawSaveDialog->FileName);
	  }
	 else RawRecordButton->Caption="Stop Raw Recording";
	}
  }
 }
 else
 {
   delete RecordRawStream;
   RecordRawStream=NULL;
   RawRecordButton->Caption="Raw Record";
 }
}
//---------------------------------------------------------------------------
void __fastcall TForm1::RawPlaybackButtonClick(TObject *Sender)
{
  if ((RawPlaybackButton->Caption=="Raw Playback") && (Sender!=NULL))
 {
  if (PlaybackRawDialog->Execute())
   {
	// First, check if the file exists.
	if (!FileExists(PlaybackRawDialog->FileName))
	  ShowMessage("File "+PlaybackRawDialog->FileName+" does not exist");
	else
	{
		// Open a file for writing. Creates the file if it doesn't exist, or overwrites it if it does.
	PlayBackRawStream= new TStreamReader(PlaybackRawDialog->FileName);
	if (PlayBackRawStream==NULL)
	  {
		ShowMessage("Cannot Open File "+PlaybackRawDialog->FileName);
	  }
	 else {
		   TCPClientRawHandleThread = new TTCPClientRawHandleThread(true);
		   TCPClientRawHandleThread->UseFileInsteadOfNetwork=true;
		   TCPClientRawHandleThread->First=true;
		   TCPClientRawHandleThread->FreeOnTerminate=TRUE;
		   TCPClientRawHandleThread->Resume();
		   RawPlaybackButton->Caption="Stop Raw Playback";
           RawConnectButton->Enabled=false;
		  }
	}
  }
 }
 else
 {
   TCPClientRawHandleThread->Terminate();
   delete PlayBackRawStream;
   PlayBackRawStream=NULL;
   RawPlaybackButton->Caption="Raw Playback";
   RawConnectButton->Enabled=true;
 }
}
//---------------------------------------------------------------------------
// Constructor for the thread class
__fastcall TTCPClientRawHandleThread::TTCPClientRawHandleThread(bool value) : TThread(value)
{
	FreeOnTerminate = true; // Automatically free the thread object after execution
}
//---------------------------------------------------------------------------
// Destructor for the thread class
__fastcall TTCPClientRawHandleThread::~TTCPClientRawHandleThread()
{
	// Clean up resources if needed
}
//---------------------------------------------------------------------------
// Execute method where the thread's logic resides
void __fastcall TTCPClientRawHandleThread::Execute(void)
{
  __int64 Time,SleepTime;
  while (!Terminated)
  {
	if (!UseFileInsteadOfNetwork)
	 {
	  try {
		   if (!Form1->IdTCPClientRaw->Connected()) Terminate();
	       StringMsgBuffer=Form1->IdTCPClientRaw->IOHandler->ReadLn();
		  }
       catch (...)
		{
		 TThread::Synchronize(StopTCPClient);
		 break;
		}

	 }
	 else
	 {
	  try
        {
         if (Form1->PlayBackRawStream->EndOfStream)
           {
            printf("End Raw Playback 1\n");
            TThread::Synchronize(StopPlayback);
            break;
           }
		 StringMsgBuffer= Form1->PlayBackRawStream->ReadLine();
         Time=StrToInt64(StringMsgBuffer);
		 if (First)
	      {
		   First=false;
		   LastTime=Time;
		  }
		 SleepTime=Time-LastTime;
		 LastTime=Time;
		 if (SleepTime>0) Sleep(SleepTime);
         if (Form1->PlayBackRawStream->EndOfStream)
           {
            printf("End Raw Playback 2\n");
            TThread::Synchronize(StopPlayback);
            break;
           }
		 StringMsgBuffer= Form1->PlayBackRawStream->ReadLine();
		}
        catch (...)
		{
         printf("Raw Playback Exception\n");
		 TThread::Synchronize(StopPlayback);
		 break;
		}
	   }
     try
      {
	   // Synchronize method to safely access UI components
	   TThread::Synchronize(HandleInput);
      }
	 catch (...)
     {
      ShowMessage("TTCPClientRawHandleThread::Execute Exception 3");
	 }
  }
}
//---------------------------------------------------------------------------
void __fastcall TTCPClientRawHandleThread::StopPlayback(void)
{
 Form1->RawPlaybackButtonClick(NULL);
}
//---------------------------------------------------------------------------
void __fastcall TTCPClientRawHandleThread::StopTCPClient(void)
{
 Form1->RawConnectButtonClick(NULL);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::CycleImagesClick(TObject *Sender)
{
 CurrentSpriteImage=0;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::SBSConnectButtonClick(TObject *Sender)
{
 IdTCPClientSBS->Host=SBSIpAddress->Text;
 IdTCPClientSBS->Port=5002;

 if ((SBSConnectButton->Caption=="SBS Connect") && (Sender!=NULL))
 {
  try
   {
   IdTCPClientSBS->Connect();
   TCPClientSBSHandleThread = new TTCPClientSBSHandleThread(true);
   TCPClientSBSHandleThread->UseFileInsteadOfNetwork=false;
   TCPClientSBSHandleThread->FreeOnTerminate=TRUE;
   TCPClientSBSHandleThread->Resume();
   }
   catch (const EIdException& e)
   {
	ShowMessage("Error while connecting: "+e.Message);
   }
 }
 else
  {
	TCPClientSBSHandleThread->Terminate();
	IdTCPClientSBS->Disconnect();
    IdTCPClientSBS->IOHandler->InputBuffer->Clear();
	SBSConnectButton->Caption="SBS Connect";
	SBSPlaybackButton->Enabled=true;
  }

}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void __fastcall TTCPClientSBSHandleThread::HandleInput(void)
{
  modeS_message mm;
  TDecodeStatus Status;

 // Form1->MsgLog->Lines->Add(StringMsgBuffer);
  if (Form1->RecordSBSStream)
  {
   __int64 CurrentTime;
   CurrentTime=GetCurrentTimeInMsec();
   Form1->RecordSBSStream->WriteLine(IntToStr(CurrentTime));
   Form1->RecordSBSStream->WriteLine(StringMsgBuffer);
  }

  if (Form1->BigQueryCSV)
  {
    Form1->BigQueryCSV->WriteLine(StringMsgBuffer);
    Form1->BigQueryRowCount++;
	if (Form1->BigQueryRowCount>=BIG_QUERY_UPLOAD_COUNT)
	{
	 Form1->CloseBigQueryCSV();
	 //printf("string is:%s\n", Form1->BigQueryPythonScript.c_str());
	 RunPythonScript(Form1->BigQueryPythonScript,Form1->BigQueryPath+" "+Form1->BigQueryCSVFileName);
	 Form1->CreateBigQueryCSV();
	}
  }
  SBS_Message_Decode( StringMsgBuffer.c_str());

}
//---------------------------------------------------------------------------
// Constructor for the thread class
__fastcall TTCPClientSBSHandleThread::TTCPClientSBSHandleThread(bool value) : TThread(value)
{
	FreeOnTerminate = true; // Automatically free the thread object after execution
}
//---------------------------------------------------------------------------
// Destructor for the thread class
__fastcall TTCPClientSBSHandleThread::~TTCPClientSBSHandleThread()
{
	// Clean up resources if needed
}
//---------------------------------------------------------------------------
// Execute method where the thread's logic resides
void __fastcall TTCPClientSBSHandleThread::Execute(void)
{
  __int64 Time,SleepTime;
  while (!Terminated)
  {
	if (!UseFileInsteadOfNetwork)
	 {
	  try {
		   if (!Form1->IdTCPClientSBS->Connected()) Terminate();
	       StringMsgBuffer=Form1->IdTCPClientSBS->IOHandler->ReadLn();
		  }
       catch (...)
		{
		 TThread::Synchronize(StopTCPClient);
		 break;
		}

	 }
	 else
	 {
	  try
        {
         if (Form1->PlayBackSBSStream->EndOfStream)
           {
            printf("End SBS Playback 1\n");
            TThread::Synchronize(StopPlayback);
            break;
           }
		 StringMsgBuffer= Form1->PlayBackSBSStream->ReadLine();
         Time=StrToInt64(StringMsgBuffer);
		 if (First)
	      {
		   First=false;
		   LastTime=Time;
		  }
		 SleepTime=Time-LastTime;
		 LastTime=Time;
		 if (SleepTime>0) Sleep(SleepTime);
         if (Form1->PlayBackSBSStream->EndOfStream)
           {
            printf("End SBS Playback 2\n");
            TThread::Synchronize(StopPlayback);
            break;
           }
		 StringMsgBuffer= Form1->PlayBackSBSStream->ReadLine();
		}
        catch (...)
		{
         printf("SBS Playback Exception\n");
		 TThread::Synchronize(StopPlayback);
		 break;
		}
	   }
     try
      {
	   // Synchronize method to safely access UI components
	   TThread::Synchronize(HandleInput);
      }
	 catch (...)
     {
      ShowMessage("TTCPClientSBSHandleThread::Execute Exception 3");
	 }
  }
}
//---------------------------------------------------------------------------
void __fastcall TTCPClientSBSHandleThread::StopPlayback(void)
{
 Form1->SBSPlaybackButtonClick(NULL);
}
//---------------------------------------------------------------------------
void __fastcall TTCPClientSBSHandleThread::StopTCPClient(void)
{
 Form1->SBSConnectButtonClick(NULL);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::SBSRecordButtonClick(TObject *Sender)
{
 if (SBSRecordButton->Caption=="SBS Record")
 {
  if (RecordSBSSaveDialog->Execute())
   {
	// First, check if the file exists.
	if (FileExists(RecordSBSSaveDialog->FileName))
	  ShowMessage("File "+RecordSBSSaveDialog->FileName+"already exists. Cannot overwrite.");
	else
	{
		// Open a file for writing. Creates the file if it doesn't exist, or overwrites it if it does.
	RecordSBSStream= new TStreamWriter(RecordSBSSaveDialog->FileName, false);
	if (RecordSBSStream==NULL)
	  {
		ShowMessage("Cannot Open File "+RecordSBSSaveDialog->FileName);
	  }
	 else SBSRecordButton->Caption="Stop SBS Recording";
	}
  }
 }
 else
 {
   delete RecordSBSStream;
   RecordSBSStream=NULL;
   SBSRecordButton->Caption="SBS Record";
 }

}
//---------------------------------------------------------------------------
void __fastcall TForm1::SBSPlaybackButtonClick(TObject *Sender)
{
  if ((SBSPlaybackButton->Caption=="SBS Playback") && (Sender!=NULL))
 {
  if (PlaybackSBSDialog->Execute())
   {
	// First, check if the file exists.
	if (!FileExists(PlaybackSBSDialog->FileName))
	  ShowMessage("File "+PlaybackSBSDialog->FileName+" does not exist");
	else
	{
		// Open a file for writing. Creates the file if it doesn't exist, or overwrites it if it does.
	PlayBackSBSStream= new TStreamReader(PlaybackSBSDialog->FileName);
	if (PlayBackSBSStream==NULL)
	  {
		ShowMessage("Cannot Open File "+PlaybackSBSDialog->FileName);
	  }
	 else {
		   TCPClientSBSHandleThread = new TTCPClientSBSHandleThread(true);
		   TCPClientSBSHandleThread->UseFileInsteadOfNetwork=true;
		   TCPClientSBSHandleThread->First=true;
		   TCPClientSBSHandleThread->FreeOnTerminate=TRUE;
		   TCPClientSBSHandleThread->Resume();
		   SBSPlaybackButton->Caption="Stop SBS Playback";
           SBSConnectButton->Enabled=false;
		  }
	}
  }
 }
 else
 {
   TCPClientSBSHandleThread->Terminate();
   delete PlayBackSBSStream;
   PlayBackSBSStream=NULL;
   SBSPlaybackButton->Caption="SBS Playback";
   SBSConnectButton->Enabled=true;
 }

}
//---------------------------------------------------------------------------

void __fastcall TForm1::IdTCPClientSBSConnected(TObject *Sender)
{
   //SetKeepAliveValues(const AEnabled: Boolean; const ATimeMS, AInterval: Integer);
   IdTCPClientSBS->Socket->Binding->SetKeepAliveValues(true,60*1000,15*1000);
   SBSConnectButton->Caption="SBS Disconnect";
   SBSPlaybackButton->Enabled=false;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::IdTCPClientSBSDisconnected(TObject *Sender)
{
  TCPClientSBSHandleThread->Terminate();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::TimeToGoTrackBarChange(TObject *Sender)
{
  _int64 hmsm;
  hmsm=TimeToGoTrackBar->Position*1000;
  TimeToGoText->Caption=TimeToChar(hmsm);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::LoadMap(int Type)
{
   AnsiString  HomeDir = ExtractFilePath(ExtractFileDir(Application->ExeName));
    if (Type==GoogleMaps)
   {
     HomeDir+= "..\\GoogleMap";
     if (LoadMapFromInternet) HomeDir+= "_Live\\";
     else  HomeDir+= "\\";
     std::string cachedir;
     cachedir=HomeDir.c_str();

     if (mkdir(cachedir.c_str()) != 0 && errno != EEXIST)
	    throw Sysutils::Exception("Can not create cache directory");

     g_Storage = new FilesystemStorage(cachedir,true);
     if (LoadMapFromInternet)
       {
	    g_Keyhole = new KeyholeConnection(GoogleMaps);
        g_Keyhole->SetSaveStorage(g_Storage);
	    g_Storage->SetNextLoadStorage(g_Keyhole);
	   }
    }
  else if (Type==SkyVector_VFR)
   {
     HomeDir+= "..\\VFR_Map";
     if (LoadMapFromInternet) HomeDir+= "_Live\\";
     else  HomeDir+= "\\";
     std::string cachedir;
     cachedir=HomeDir.c_str();

     if (mkdir(cachedir.c_str()) != 0 && errno != EEXIST)
	    throw Sysutils::Exception("Can not create cache directory");

     g_Storage = new FilesystemStorage(cachedir,true);
     if (LoadMapFromInternet)
       {
	    g_Keyhole = new KeyholeConnection(SkyVector_VFR);
        g_Keyhole->SetSaveStorage(g_Storage);
	    g_Storage->SetNextLoadStorage(g_Keyhole);
	   }
    }
  else if (Type==SkyVector_IFR_Low)
   {
     HomeDir+= "..\\IFR_Low_Map";
     if (LoadMapFromInternet) HomeDir+= "_Live\\";
     else  HomeDir+= "\\";
     std::string cachedir;
     cachedir=HomeDir.c_str();

     if (mkdir(cachedir.c_str()) != 0 && errno != EEXIST)
	    throw Sysutils::Exception("Can not create cache directory");

     g_Storage = new FilesystemStorage(cachedir,true);
     if (LoadMapFromInternet)
       {
	    g_Keyhole = new KeyholeConnection(SkyVector_IFR_Low);
        g_Keyhole->SetSaveStorage(g_Storage);
	    g_Storage->SetNextLoadStorage(g_Keyhole);
	   }
    }
  else if (Type==SkyVector_IFR_High)
   {
     HomeDir+= "..\\IFR_High_Map";
     if (LoadMapFromInternet) HomeDir+= "_Live\\";
     else  HomeDir+= "\\";
     std::string cachedir;
     cachedir=HomeDir.c_str();

     if (mkdir(cachedir.c_str()) != 0 && errno != EEXIST)
	    throw Sysutils::Exception("Can not create cache directory");

     g_Storage = new FilesystemStorage(cachedir,true);
     if (LoadMapFromInternet)
       {
	    g_Keyhole = new KeyholeConnection(SkyVector_IFR_High);
        g_Keyhole->SetSaveStorage(g_Storage);
	    g_Storage->SetNextLoadStorage(g_Keyhole);
	   }
    }
   g_GETileManager = new TileManager(g_Storage);
   g_MasterLayer = new GoogleLayer(g_GETileManager);

   g_EarthView = new FlatEarthView(g_MasterLayer);
   g_EarthView->Resize(ObjectDisplay->Width,ObjectDisplay->Height);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::MapComboBoxChange(TObject *Sender)
{
  double    m_Eyeh= g_EarthView->m_Eye.h;
  double    m_Eyex= g_EarthView->m_Eye.x;
  double    m_Eyey= g_EarthView->m_Eye.y;


  Timer1->Enabled=false;
  Timer2->Enabled=false;
  delete g_EarthView;
  if (g_GETileManager) delete g_GETileManager;
  delete g_MasterLayer;
  delete g_Storage;
  if (LoadMapFromInternet)
  {
   if (g_Keyhole) delete g_Keyhole;
  }
  if (MapComboBox->ItemIndex==0)   LoadMap(GoogleMaps);

  else if (MapComboBox->ItemIndex==1)  LoadMap(SkyVector_VFR);

  else if (MapComboBox->ItemIndex==2)  LoadMap(SkyVector_IFR_Low);

  else if (MapComboBox->ItemIndex==3)   LoadMap(SkyVector_IFR_High);

   g_EarthView->m_Eye.h =m_Eyeh;
   g_EarthView->m_Eye.x=m_Eyex;
   g_EarthView->m_Eye.y=m_Eyey;
   Timer1->Enabled=true;
   Timer2->Enabled=true;

}
//---------------------------------------------------------------------------

void __fastcall TForm1::BigQueryCheckBoxClick(TObject *Sender)
{
 if (BigQueryCheckBox->State==cbChecked) CreateBigQueryCSV();
 else {
	   CloseBigQueryCSV();
	   RunPythonScript(BigQueryPythonScript,BigQueryPath+" "+BigQueryCSVFileName);
	  }
}
//---------------------------------------------------------------------------
void __fastcall TForm1::CreateBigQueryCSV(void)
{
    AnsiString  HomeDir = ExtractFilePath(ExtractFileDir(Application->ExeName));
    BigQueryCSVFileName="BigQuery"+UIntToStr(BigQueryFileCount)+".csv";
    BigQueryRowCount=0;
    BigQueryFileCount++;
    BigQueryCSV=new TStreamWriter(HomeDir+"..\\BigQuery\\"+BigQueryCSVFileName, false);
    if (BigQueryCSV==NULL)
	  {
		ShowMessage("Cannot Open BigQuery CSV File "+HomeDir+"..\\BigQuery\\"+BigQueryCSVFileName);
        BigQueryCheckBox->State=cbUnchecked;
	  }
	AnsiString Header=AnsiString("Message Type,Transmission Type,SessionID,AircraftID,HexIdent,FlightID,Date_MSG_Generated,Time_MSG_Generated,Date_MSG_Logged,Time_MSG_Logged,Callsign,Altitude,GroundSpeed,Track,Latitude,Longitude,VerticalRate,Squawk,Alert,Emergency,SPI,IsOnGround");
	BigQueryCSV->WriteLine(Header);
}
//--------------------------------------------------------------------------
void __fastcall TForm1::CloseBigQueryCSV(void)
{
    if (BigQueryCSV)
    {
     delete BigQueryCSV;
     BigQueryCSV=NULL;
    }
}
//--------------------------------------------------------------------------
	 static void RunPythonScript(AnsiString scriptPath,AnsiString args)
     {
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;

        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        AnsiString commandLine = "python " + scriptPath+" "+args;
        char* cmdLineCharArray = new char[strlen(commandLine.c_str()) + 1];
		strcpy(cmdLineCharArray, commandLine.c_str());
	#define  LOG_PYTHON 1
	#if LOG_PYTHON
        //printf("%s\n", cmdLineCharArray);
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(sa);
	    sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE;
		HANDLE h = CreateFileA(Form1->BigQueryLogFileName.c_str(),
		FILE_APPEND_DATA,
        FILE_SHARE_WRITE | FILE_SHARE_READ,
        &sa,
		OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
		NULL );

        si.hStdInput = NULL;
	    si.hStdOutput = h;
	    si.hStdError = h; // Redirect standard error as well, if needed
	    si.dwFlags |= STARTF_USESTDHANDLES;
    #endif
        if (!CreateProcessA(
            nullptr,          // No module name (use command line)
            cmdLineCharArray, // Command line
            nullptr,          // Process handle not inheritable
            nullptr,          // Thread handle not inheritable
	 #if LOG_PYTHON
            TRUE,
     #else
            FALSE,            // Set handle inheritance to FALSE
     #endif
            CREATE_NO_WINDOW, // Don't create a console window
			nullptr,          // Use parent's environment block
            nullptr,          // Use parent's starting directory
            &si,             // Pointer to STARTUPINFO structure
            &pi))             // Pointer to PROCESS_INFORMATION structure
         {
            std::cerr << "CreateProcess failed (" << GetLastError() << ").\n";
            delete[] cmdLineCharArray;
            return;
         }

        // Optionally, detach from the process
        CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
		delete[] cmdLineCharArray;
    }

//---------------------------------------------------------------------------
// Run Python script and capture output
//---------------------------------------------------------------------------
static AnsiString RunPythonScriptWithOutput(AnsiString scriptPath, AnsiString args)
{
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	HANDLE hStdOutRd, hStdOutWr;
	SECURITY_ATTRIBUTES sa;
	char buffer[4096];
	DWORD bytesRead;
	AnsiString output = "";

	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));
	ZeroMemory(&sa, sizeof(sa));
	sa.nLength = sizeof(sa);
	sa.bInheritHandle = TRUE;
	sa.lpSecurityDescriptor = NULL;

	// Create pipe for stdout
	if (!CreatePipe(&hStdOutRd, &hStdOutWr, &sa, 0))
	{
		return "ERROR: Failed to create pipe";
	}

	si.hStdOutput = hStdOutWr;
	si.hStdError = hStdOutWr;
	si.dwFlags |= STARTF_USESTDHANDLES;

	AnsiString commandLine = "python \"" + scriptPath + "\" " + args;
	char* cmdLineCharArray = new char[strlen(commandLine.c_str()) + 1];
	strcpy(cmdLineCharArray, commandLine.c_str());

	if (CreateProcessA(
		nullptr,
		cmdLineCharArray,
		nullptr,
		nullptr,
		TRUE,
		CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&si,
		&pi))
	{
		CloseHandle(hStdOutWr);

		// Wait for process with timeout
		DWORD waitResult = WaitForSingleObject(pi.hProcess, 15000);
		
		if (waitResult == WAIT_OBJECT_0)
		{
			// Read output
			while (ReadFile(hStdOutRd, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0)
			{
				buffer[bytesRead] = '\0';
				output += AnsiString(buffer);
			}
		}
		else
		{
			TerminateProcess(pi.hProcess, 1);
			output = "ERROR: Timeout";
		}

		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
	}
	else
	{
		output = "ERROR: Failed to start process";
	}

	CloseHandle(hStdOutRd);
	delete[] cmdLineCharArray;
	
	return output.Trim();
}

 //--------------------------------------------------------------------------
void __fastcall TForm1::UseSBSRemoteClick(TObject *Sender)
{
 SBSIpAddress->Text="data.adsbhub.org";
}
//---------------------------------------------------------------------------

void __fastcall TForm1::UseSBSLocalClick(TObject *Sender)
{
 SBSIpAddress->Text="128.237.96.41";
}
//---------------------------------------------------------------------------
// Audio Recorder Implementation
//---------------------------------------------------------------------------
__fastcall TAudioRecorder::TAudioRecorder()
{
	hWaveIn = NULL;
	waveHeaders = NULL;
	numBuffers = 3;
	isRecording = false;
	outputFilePath = "";
	
	// Silence detection parameters
	silenceThreshold = 0.01;  // RMS threshold (1% of max amplitude)
	silenceDurationMs = 2000; // 2 seconds of silence before auto-stop
	currentSilenceMs = 0;
	lastSoundTime = 0;
	autoStopEnabled = true;
	autoStopCallback = NULL;
	
	// Initialize WAV format: 16kHz, 16-bit, mono (optimal for Whisper)
	ZeroMemory(&wfx, sizeof(::WAVEFORMATEX));
	wfx.wFormatTag = WAVE_FORMAT_PCM;
	wfx.nChannels = 1;  // Mono
	wfx.nSamplesPerSec = 16000;  // 16kHz (Whisper's preferred sample rate)
	wfx.wBitsPerSample = 16;
	wfx.nBlockAlign = wfx.nChannels * wfx.wBitsPerSample / 8;
	wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
	wfx.cbSize = 0;
}

__fastcall TAudioRecorder::~TAudioRecorder()
{
	StopRecording();
}

void CALLBACK TAudioRecorder::WaveInProc(::HWAVEIN hWaveIn, UINT uMsg, ::DWORD_PTR dwInstance, ::DWORD_PTR dwParam1, ::DWORD_PTR dwParam2)
{
	TAudioRecorder* pRecorder = (TAudioRecorder*)dwInstance;
	
	if (uMsg == WIM_DATA)
	{
		::WAVEHDR* pwh = (::WAVEHDR*)dwParam1;
		if (pRecorder && pRecorder->isRecording)
		{
			pRecorder->ProcessBuffer(pwh);
		}
	}
}

double TAudioRecorder::CalculateRMS(short* samples, int numSamples)
{
	if (numSamples == 0) return 0.0;
	
	double sum = 0.0;
	for (int i = 0; i < numSamples; i++)
	{
		double normalized = (double)samples[i] / 32768.0; // Normalize to -1.0 to 1.0
		sum += normalized * normalized;
	}
	return sqrt(sum / numSamples);
}

void TAudioRecorder::CheckSilenceAndAutoStop()
{
	if (!autoStopEnabled || !isRecording)
		return;
	
	DWORD currentTime = GetTickCount();
	
	// Check if we need to reset silence timer (first sound after silence)
	if (lastSoundTime == 0)
	{
		lastSoundTime = currentTime;
		return;
	}
	
	// Calculate silence duration
	currentSilenceMs = currentTime - lastSoundTime;
	
	// Auto-stop if silence exceeds threshold
	if (currentSilenceMs >= silenceDurationMs && isRecording)
	{
		printf("WHISPER: Auto-stop detected (silence for %d ms)\n", currentSilenceMs);
		
		// Set flag first to prevent re-entry (don't call StopRecording here - do it in main thread)
		isRecording = false;
		
		printf("WHISPER: Posting message to main thread for safe UI update...\n");
		
		// Notify callback (TForm1*) - use PostMessage for async processing to avoid deadlock
		if (autoStopCallback)
		{
			TForm1* form = static_cast<TForm1*>(autoStopCallback);
			// PostMessage for async queuing - main thread handler will call StopRecording()
			BOOL result = PostMessage(form->Handle, WM_AUTO_STOP_RECORDING, 0, 0);
			printf("WHISPER: PostMessage result: %d, Handle: %p\n", result, form->Handle);
		}
		else
		{
			printf("WHISPER ERROR: autoStopCallback is NULL!\n");
		}
	}
}

void TAudioRecorder::ProcessBuffer(::WAVEHDR* pwh)
{
	// Check if we're still recording (might have been stopped)
	if (!isRecording)
	{
		printf("TAudioRecorder::ProcessBuffer() - not recording, skipping\n");
		return;
	}
	
	// Calculate audio level for silence detection
	if (pwh->dwBytesRecorded > 0 && autoStopEnabled)
	{
		int numSamples = pwh->dwBytesRecorded / sizeof(short);
		short* samples = (short*)pwh->lpData;
		double rms = CalculateRMS(samples, numSamples);
		
		// Check if sound is above threshold
		if (rms > silenceThreshold)
		{
			// Sound detected - reset silence timer
			lastSoundTime = GetTickCount();
			currentSilenceMs = 0;
		}
		else
		{
			// Silence detected - check if we should auto-stop
			CheckSilenceAndAutoStop();
		}
	}
	
	// Write buffer to file (only if still recording)
	if (isRecording && outputFilePath.Length() > 0 && pwh->dwBytesRecorded > 0)
	{
		FILE* fp = fopen(outputFilePath.c_str(), "ab");
		if (fp)
		{
			fwrite(pwh->lpData, 1, pwh->dwBytesRecorded, fp);
			fclose(fp);
		}
	}
	
	// Re-add buffer to queue if still recording
	if (isRecording && hWaveIn)
	{
		waveInAddBuffer(hWaveIn, pwh, sizeof(::WAVEHDR));
	}
	else
	{
		printf("TAudioRecorder::ProcessBuffer() - not re-adding buffer (isRecording=%d, hWaveIn=%p)\n", isRecording, hWaveIn);
	}
}

bool TAudioRecorder::StartRecording(AnsiString filePath, bool enableAutoStop)
{
	if (isRecording)
	{
		return false; // Already recording
	}
	
	outputFilePath = filePath;
	autoStopEnabled = enableAutoStop;
	currentSilenceMs = 0;
	lastSoundTime = 0; // Reset silence timer
	
	// Delete existing file
	DeleteFileA(filePath.c_str());
	
	// Open wave input device
	MMRESULT result = waveInOpen(&hWaveIn, WAVE_MAPPER, &wfx, (::DWORD_PTR)WaveInProc, (::DWORD_PTR)this, CALLBACK_FUNCTION);
	if (result != MMSYSERR_NOERROR)
	{
		return false;
	}
	
	// Allocate buffers
	waveHeaders = new ::WAVEHDR[numBuffers];
	int bufferSize = wfx.nAvgBytesPerSec / 2; // 0.5 second buffers
	
	for (int i = 0; i < numBuffers; i++)
	{
		ZeroMemory(&waveHeaders[i], sizeof(::WAVEHDR));
		waveHeaders[i].lpData = new char[bufferSize];
		waveHeaders[i].dwBufferLength = bufferSize;
		waveInPrepareHeader(hWaveIn, &waveHeaders[i], sizeof(::WAVEHDR));
		waveInAddBuffer(hWaveIn, &waveHeaders[i], sizeof(::WAVEHDR));
	}
	
	// Write WAV header
	FILE* fp = fopen(filePath.c_str(), "wb");
	if (!fp)
	{
		StopRecording();
		return false;
	}
	
	// WAV header structure
	struct WAVHeader
	{
		char riff[4] = {'R', 'I', 'F', 'F'};
		DWORD chunkSize = 0; // Will be filled later
		char wave[4] = {'W', 'A', 'V', 'E'};
		char fmt[4] = {'f', 'm', 't', ' '};
		DWORD fmtSize = 16;
		WORD audioFormat = 1; // PCM
		WORD numChannels = 1;
		DWORD sampleRate = 16000;
		DWORD byteRate = 32000;
		WORD blockAlign = 2;
		WORD bitsPerSample = 16;
		char data[4] = {'d', 'a', 't', 'a'};
		DWORD dataSize = 0; // Will be filled later
	} header;
	
	fwrite(&header, 1, sizeof(header), fp);
	fclose(fp);
	
	// Start recording
	result = waveInStart(hWaveIn);
	if (result != MMSYSERR_NOERROR)
	{
		StopRecording();
		return false;
	}
	
	isRecording = true;
	return true;
}

void TAudioRecorder::StopRecording()
{
	if (!isRecording && !hWaveIn)
	{
		return;
	}
	
	isRecording = false;
	
	if (hWaveIn)
	{
		waveInStop(hWaveIn);
		waveInReset(hWaveIn);
		
		// Unprepare headers
		if (waveHeaders)
		{
			for (int i = 0; i < numBuffers; i++)
			{
				waveInUnprepareHeader(hWaveIn, &waveHeaders[i], sizeof(::WAVEHDR));
				delete[] waveHeaders[i].lpData;
			}
			delete[] waveHeaders;
			waveHeaders = NULL;
		}
		
		waveInClose(hWaveIn);
		hWaveIn = NULL;
	}
	
	// Update WAV file header with actual sizes
	if (outputFilePath.Length() > 0)
	{
		FILE* fp = fopen(outputFilePath.c_str(), "r+b");
		if (fp)
		{
			fseek(fp, 0, SEEK_END);
			long fileSize = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			
			// Update chunk sizes
			DWORD chunkSize = fileSize - 8;
			DWORD dataSize = fileSize - 44; // Subtract header size
			
			fseek(fp, 4, SEEK_SET);
			fwrite(&chunkSize, 4, 1, fp);
			fseek(fp, 40, SEEK_SET);
			fwrite(&dataSize, 4, 1, fp);
			
			fclose(fp);
		}
	}
}
//---------------------------------------------------------------------------
// Speech Transcription Thread Implementation
//---------------------------------------------------------------------------
__fastcall TSpeechTranscribeThread::TSpeechTranscribeThread(AnsiString audioFile, AnsiString pythonPath, AnsiString scriptPath) : TThread(true)
{
	AudioFilePath = audioFile;
	PythonPath = pythonPath;
	TranscribeScriptPath = scriptPath;
	TranscribedText = "";
	TranscriptionSuccess = false;
	FreeOnTerminate = false;
}

__fastcall TSpeechTranscribeThread::~TSpeechTranscribeThread()
{
}

void __fastcall TSpeechTranscribeThread::Execute(void)
{
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	HANDLE hStdOutRd, hStdOutWr;
	SECURITY_ATTRIBUTES sa;
	char buffer[4096];
	DWORD bytesRead;
	AnsiString output = "";

	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));
	ZeroMemory(&sa, sizeof(sa));
	sa.nLength = sizeof(sa);
	sa.bInheritHandle = TRUE;
	sa.lpSecurityDescriptor = NULL;

	// Create pipe for stdout
	if (!CreatePipe(&hStdOutRd, &hStdOutWr, &sa, 0))
	{
		TranscriptionSuccess = false;
		TranscribedText = "ERROR: Failed to create pipe";
		return;
	}

	// Create separate pipe for stderr to avoid mixing with stdout
	HANDLE hStdErrRd, hStdErrWr;
	if (!CreatePipe(&hStdErrRd, &hStdErrWr, &sa, 0))
	{
		CloseHandle(hStdOutRd);
		CloseHandle(hStdOutWr);
		TranscriptionSuccess = false;
		TranscribedText = "ERROR: Failed to create stderr pipe";
		return;
	}

	si.hStdOutput = hStdOutWr;
	si.hStdError = hStdErrWr;  // Separate stderr
	si.dwFlags |= STARTF_USESTDHANDLES;

	// Build command line: python.exe script_path audio_file --ollama --bigquery --max-rows 5
	// --ollama flag sends Whisper transcription to Ollama LLM to generate SQL
	// --bigquery flag executes the SQL on BigQuery and returns results
	// --max-rows 5 displays only top 5 results in GUI panel
	// This will show: transcription → SQL → BigQuery results (top 5)
	AnsiString commandLine = "\"" + PythonPath + "\" \"" + TranscribeScriptPath + "\" \"" + AudioFilePath + "\" --ollama --bigquery --max-rows 5";
	
	// Debug: Print Whisper + Ollama connection info
	printf("=== WHISPER + OLLAMA CONNECTION DEBUG ===\n");
	printf("Python Path: %s\n", PythonPath.c_str());
	printf("Script Path: %s\n", TranscribeScriptPath.c_str());
	printf("Audio File: %s\n", AudioFilePath.c_str());
	printf("Command: %s\n", commandLine.c_str());
	printf("==========================================\n");
	
	char* cmdLineCharArray = new char[strlen(commandLine.c_str()) + 1];
	strcpy(cmdLineCharArray, commandLine.c_str());

	if (CreateProcessA(
		nullptr,
		cmdLineCharArray,
		nullptr,
		nullptr,
		TRUE,
		CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&si,
		&pi))
	{
		CloseHandle(hStdOutWr);
		CloseHandle(hStdErrWr);

		// Read output with timeout (90 seconds - Whisper + Ollama needs more time)
		DWORD waitResult = WaitForSingleObject(pi.hProcess, 90000);
		
		if (waitResult == WAIT_OBJECT_0)
		{
			// Read stdout (transcription result)
			while (ReadFile(hStdOutRd, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0)
			{
				buffer[bytesRead] = '\0';
				output += AnsiString(buffer);
				// Debug: Print raw output from Whisper
				printf("WHISPER stdout: %s", buffer);
			}
			
			// Read stderr (error messages) - but don't use as transcription
			AnsiString errorOutput = "";
			while (ReadFile(hStdErrRd, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0)
			{
				buffer[bytesRead] = '\0';
				errorOutput += AnsiString(buffer);
				// Debug: Print errors from Whisper
				printf("WHISPER stderr: %s", buffer);
			}
			
			// Check exit code
			DWORD exitCode = 0;
			GetExitCodeProcess(pi.hProcess, &exitCode);
			
			// Trim whitespace
			output = output.Trim();
			
			if (exitCode == 0 && output.Length() > 0)
			{
				TranscribedText = output;
				TranscriptionSuccess = true;
			}
			else
			{
				TranscriptionSuccess = false;
				if (errorOutput.Length() > 0)
				{
					TranscribedText = "ERROR: " + errorOutput;
				}
				else if (output.Length() == 0)
				{
					TranscribedText = "ERROR: No transcription output. Audio may be empty or model not loaded. Please try again.";
				}
				else
				{
					TranscribedText = "ERROR: Transcription failed (exit code: " + AnsiString((int)exitCode) + ")";
				}
			}
		}
		else
		{
			// Timeout or error
			TerminateProcess(pi.hProcess, 1);
			TranscriptionSuccess = false;
			TranscribedText = "ERROR: Transcription timeout (90s). Whisper/Ollama may be loading - please try again.";
		}

		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
	}
	else
	{
		CloseHandle(hStdOutWr);
		CloseHandle(hStdErrWr);
		TranscriptionSuccess = false;
		TranscribedText = "ERROR: Failed to start transcription process. Check Python path.";
	}

	CloseHandle(hStdOutRd);
	CloseHandle(hStdErrRd);
	delete[] cmdLineCharArray;

	// Update UI in main thread
	Synchronize(HandleTranscriptionResult);
}

void __fastcall TSpeechTranscribeThread::HandleTranscriptionResult(void)
{
	if (!Form1) return;
	
	// Ensure recording flag is cleared
	Form1->IsRecordingVoice = false;
	
	// Add transcription result to Memo1
	if (TranscriptionSuccess && TranscribedText.Length() > 0)
	{
		if (Form1->Memo1)
		{
			Form1->Memo1->Lines->Add(TranscribedText);
		}
		printf("WHISPER: Transcription added to Memo1: %s\n", TranscribedText.c_str());
		// Optionally process voice command here: Form1->ProcessVoiceCommand(TranscribedText);
	}
	else
	{
		if (Form1->Memo1)
		{
			Form1->Memo1->Lines->Add("Recognition failed: " + TranscribedText);
		}
		printf("WHISPER: Transcription failed: %s\n", TranscribedText.c_str());
	}
}
//---------------------------------------------------------------------------
// Speech Transcription Functions
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void __fastcall TForm1::ProcessVoiceCommand(AnsiString command)
{
	// Convert to lowercase for case-insensitive matching
	AnsiString cmd = command.LowerCase().Trim();
	AnsiString recognizedCommand = "";
	
	// Command mapping table
	// Zoom commands
	if (stristr(cmd.c_str(), "zoom in") != NULL)
	{
		ZoomInClick(NULL);
		recognizedCommand = "Zoom In";
	}
	else if (stristr(cmd.c_str(), "zoom out") != NULL)
	{
		ZoomOutClick(NULL);
		recognizedCommand = "Zoom Out";
	}
	// Map toggle
	else if (stristr(cmd.c_str(), "map") != NULL)
	{
		if (stristr(cmd.c_str(), "on") != NULL || stristr(cmd.c_str(), "show") != NULL)
		{
			DrawMap->Checked = true;
			recognizedCommand = "Map On";
		}
		else if (stristr(cmd.c_str(), "off") != NULL || stristr(cmd.c_str(), "hide") != NULL)
		{
			DrawMap->Checked = false;
			recognizedCommand = "Map Off";
		}
	}
	// Traffic filter
	else if (stristr(cmd.c_str(), "traffic") != NULL)
	{
		if (stristr(cmd.c_str(), "filter") != NULL)
		{
			if (stristr(cmd.c_str(), "on") != NULL || stristr(cmd.c_str(), "enable") != NULL)
			{
				PurgeStale->Checked = true;
				recognizedCommand = "Traffic Filter On";
			}
			else if (stristr(cmd.c_str(), "off") != NULL || stristr(cmd.c_str(), "disable") != NULL)
			{
				PurgeStale->Checked = false;
				recognizedCommand = "Traffic Filter Off";
			}
		}
	}
	// Aircraft selection (e.g., "select N12345")
	else if (stristr(cmd.c_str(), "select") != NULL)
	{
		// Try to extract aircraft callsign (pattern: N followed by numbers/letters)
		const char* nPattern = stristr(cmd.c_str(), "n");
		if (nPattern != NULL)
		{
			// Extract potential callsign
			AnsiString callsign = "";
			const char* p = nPattern;
			int len = 0;
			while ((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '-')
			{
				callsign += *p;
				p++;
				len++;
				if (len > 10) break; // Limit callsign length
			}
			
			if (callsign.Length() > 1)
			{
				// TODO: Implement aircraft selection by callsign
				recognizedCommand = "Select Aircraft: " + callsign.UpperCase();
			}
		}
	}
	
	// Update UI
	if (recognizedCommand.Length() > 0)
	{
		wchar_t *wtext = AnsiTowchar_t(recognizedCommand);
		SpVoice1->Speak(wtext, SpeechVoiceSpeakFlags::SVSFlagsAsync); // Speak the recognized command
		delete wtext;
	}
	
}
//---------------------------------------------------------------------------
static bool DeleteFilesWithExtension(AnsiString dirPath, AnsiString extension)
 {
	AnsiString searchPattern = dirPath + "\\*." + extension;
	WIN32_FIND_DATAA findData;

	HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);
    #define INVALID_HANDLE_VALUE_XX ((HANDLE)(::LONG_PTR)-1)
	if (hFind == INVALID_HANDLE_VALUE_XX) {
		return false; // No files found or error
	}

	do {
		AnsiString filePath = dirPath + "\\" + findData.cFileName;
		if (DeleteFileA(filePath.c_str()) == 0) {
			FindClose(hFind);
			return false; // Failed to delete a file
		}
	} while (FindNextFileA(hFind, &findData) != 0);

	FindClose(hFind);
	return true;
}
static bool IsFirstRow=true;
static bool CallBackInit=false;
//---------------------------------------------------------------------------
 static int CSV_callback_ARTCCBoundaries (struct CSV_context *ctx, const char *value)
{
  int    rc = 1;
  static char LastArea[512];
  static char Area[512];
  static char Lat[512];
  static char Lon[512];
  int    Deg,Min,Sec,Hsec;
  char   Dir;

   if (ctx->field_num==0)
   {
	strcpy(Area,value);
   }
   else if (ctx->field_num==3)
   {
	strcpy(Lat,value);
   }
   else if (ctx->field_num==4)
   {
    strcpy(Lon,value);
   }

   if (ctx->field_num == (ctx->num_fields - 1))
   {

	float fLat, fLon;
   if (!IsFirstRow)
   {
	 if (!CallBackInit)
	 {
	  strcpy(LastArea,Area);
	  CallBackInit=true;
	 }
	   if(strcmp(LastArea,Area)!=0)
		{

		 if (FinshARTCCBoundary())
		   {
			printf("Load ERROR ID %s\n",LastArea);
		   }
		 else printf("Loaded ID %s\n",LastArea);
		 strcpy(LastArea,Area);
		 }
	   if (Form1->AreaTemp==NULL)
		   {
			Form1->AreaTemp= new TArea;
			Form1->AreaTemp->NumPoints=0;
			Form1->AreaTemp->Name=Area;
			Form1->AreaTemp->Selected=false;
			Form1->AreaTemp->Triangles=NULL;
			 printf("Loading ID %s\n",Area);
		   }
	   if (sscanf(Lat,"%2d%2d%2d%2d%c",&Deg,&Min,&Sec,&Hsec,&Dir)!=5)
		 printf("Latitude Parse Error\n");
	   fLat=Deg+Min/60.0+Sec/3600.0+Hsec/360000.00;
	   if (Dir=='S') fLat=-fLat;

	   if (sscanf(Lon,"%3d%2d%2d%2d%c",&Deg,&Min,&Sec,&Hsec,&Dir)!=5)
		 printf("Longitude Parse Error\n");
	   fLon=Deg+Min/60.0+Sec/3600.0+Hsec/360000.00;
	   if (Dir=='W') fLon=-fLon;
	   //printf("%f, %f\n",fLat,fLon);
	   if (Form1->AreaTemp->NumPoints<MAX_AREA_POINTS)
	   {
		Form1->AreaTemp->Points[Form1->AreaTemp->NumPoints][1]=fLat;
		Form1->AreaTemp->Points[Form1->AreaTemp->NumPoints][0]=fLon;
		Form1->AreaTemp->Points[Form1->AreaTemp->NumPoints][2]=0.0;
		Form1->AreaTemp->NumPoints++;
	   }
		else printf("Max Area Points Reached\n");

   }
   if (IsFirstRow) IsFirstRow=false;
   }
  return(rc);
}
//---------------------------------------------------------------------------
bool __fastcall TForm1::LoadARTCCBoundaries(AnsiString FileName)
{
  CSV_context  csv_ctx;
   memset (&csv_ctx, 0, sizeof(csv_ctx));
   csv_ctx.file_name = FileName.c_str();
   csv_ctx.delimiter = ',';
   csv_ctx.callback  = CSV_callback_ARTCCBoundaries;
   csv_ctx.line_size = 2000;
   IsFirstRow=true;
   CallBackInit=false;
   if (!CSV_open_and_parse_file(&csv_ctx))
    {
	  printf("Parsing of \"%s\" failed: %s\n", FileName.c_str(), strerror(errno));
      return (false);
	}
   if ((Form1->AreaTemp!=NULL) && (Form1->AreaTemp->NumPoints>0))
   {
     char Area[512];
     strcpy(Area,Form1->AreaTemp->Name.c_str());
     if (FinshARTCCBoundary())
	    {
        printf("Loaded ERROR ID %s\n",Area);
	    }
        else printf("Loaded ID %s\n",Area);
   }
   printf("Done\n");
   return(true);
}
//---------------------------------------------------------------------------
void __fastcall TForm1::LoadARTCCBoundaries1Click(TObject *Sender)
{
   LoadARTCCBoundaries(ARTCCBoundaryDataPathFileName);
}
//---------------------------------------------------------------------------
static int FinshARTCCBoundary(void)
{
  int or1=orientation2D_Polygon( Form1->AreaTemp->Points,Form1->AreaTemp->NumPoints);
  if (or1==0)
   {
	TArea *Temp;
	Temp= Form1->AreaTemp;
	Form1->AreaTemp=NULL;
	delete  Temp;
	printf("Degenerate Polygon\n");
	return(-1);
   }
  if (or1==CLOCKWISE)
  {
	DWORD i;

	memcpy(Form1->AreaTemp->PointsAdj,Form1->AreaTemp->Points,sizeof(Form1->AreaTemp->Points));
	for (i = 0; i <Form1->AreaTemp->NumPoints; i++)
	 {
	   memcpy(Form1->AreaTemp->Points[i],
			 Form1->AreaTemp->PointsAdj[Form1->AreaTemp->NumPoints-1-i],sizeof( pfVec3));
	 }
  }
  if (checkComplex( Form1->AreaTemp->Points,Form1->AreaTemp->NumPoints))
   {
	TArea *Temp;
	Temp= Form1->AreaTemp;
	Form1->AreaTemp=NULL;
	delete  Temp;
	printf("Polygon is Complex\n");
    return(-2);
   }
  DWORD Row,Count,i;


 Count=Form1->Areas->Count;
 for (i = 0; i < Count; i++)
 {
  TArea *Area = (TArea *)Form1->Areas->Items[i];
  if (Area->Name==Form1->AreaTemp->Name) {

   TArea *Temp;
   Temp= Form1->AreaTemp;
   printf("Duplicate Area Name %s\n",Form1->AreaTemp->Name.c_str());;
   Form1->AreaTemp=NULL;
   delete  Temp;
   return(-3);
   }
 }

 triangulatePoly(Form1->AreaTemp->Points,Form1->AreaTemp->NumPoints,
				 &Form1->AreaTemp->Triangles);

 Form1->AreaTemp->Color=TColor(PopularColors[CurrentColor]);
 CurrentColor++ ;
 CurrentColor=CurrentColor%NumColors;
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
 return 0 ;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::SpSharedRecoContext1Recognition(TObject *Sender, long StreamNumber,
          Variant StreamPosition, SpeechRecognitionType RecognitionType,
          ISpeechRecoResult *Result)
{
 // SAPI recognition disabled - using Whisper instead
 // Whisper transcription results go directly to Memo1 via TSpeechTranscribeThread::HandleTranscriptionResult
}
//---------------------------------------------------------------------------

// Helper function to start transcription
void TForm1::StartTranscription()
{
	printf("WHISPER: Starting transcription...\n");
	
	// Ensure Memo1 is visible
	if (Memo1)
	{
		Memo1->Visible = true;
	}
	
	// Get the recorded audio file path
	if (RecordedAudioPath.Length() == 0)
	{
		AnsiString HomeDir = ExtractFilePath(ExtractFileDir(Application->ExeName));
		RecordedAudioPath = HomeDir + AnsiString("..\\..\\speech\\temp_recording.wav");
	}
	
	// Verify file exists and has content
	FILE* fp = fopen(RecordedAudioPath.c_str(), "rb");
	if (fp)
	{
		fseek(fp, 0, SEEK_END);
		long fileSize = ftell(fp);
		fclose(fp);
		
		if (fileSize > 44) // More than just WAV header
		{
			// Start transcription thread
			if (SpeechTranscribeThread)
			{
				// Wait for previous thread to finish if still running
				if (!SpeechTranscribeThread->Finished)
				{
					SpeechTranscribeThread->Terminate();
					SpeechTranscribeThread->WaitFor();
				}
				delete SpeechTranscribeThread;
			}
			SpeechTranscribeThread = new TSpeechTranscribeThread(
				RecordedAudioPath,
				SpeechPythonPath,
				SpeechTranscribeScriptPath
			);
			SpeechTranscribeThread->Start();
		}
		else
		{
			if (Memo1)
			{
				Memo1->Lines->Add("Error: No recorded audio");
			}
			printf("WHISPER ERROR: No recorded audio\n");
		}
	}
	else
	{
		if (Memo1)
		{
			Memo1->Lines->Add("Error: Audio file not found");
		}
		printf("WHISPER ERROR: Audio file not found\n");
	}
}

// Message handler for auto-stop recording
void __fastcall TForm1::WMAutoStopRecording(TMessage &Message)
{
	printf("WHISPER: WMAutoStopRecording message received!\n");
	
	// Called when recording auto-stops due to silence
	// Stop recording safely in main thread, then start transcription
	
	// Stop recording first - safely in main thread
	if (AudioRecorder && AudioRecorder->IsRecording())
	{
		printf("WHISPER: Stopping recording from main thread...\n");
		AudioRecorder->StopRecording();
	}
	
	IsRecordingVoice = false;
	
	// User feedback
	if (Memo1)
	{
		Memo1->Visible = true;
		Memo1->Lines->Add("Recording auto-stopped");
	}
	else
	{
		printf("WHISPER ERROR: Memo1 is NULL!\n");
	}
	
	// Start transcription immediately
	printf("WHISPER: Starting transcription...\n");
	StartTranscription();
	
	Message.Result = 0;
}

void __fastcall TForm1::LIstenClick(TObject *Sender)
{
	printf("=== WHISPER LISTEN CLICKED ===\n");
	
	Memo1->Visible = true;
	
	if (IsRecordingVoice)
	{
		// Manual stop - same logic as auto-stop for consistency
		if (AudioRecorder && AudioRecorder->IsRecording())
		{
			AudioRecorder->StopRecording();
		}
		IsRecordingVoice = false;
		
		// User feedback
		if (Memo1)
		{
			Memo1->Lines->Add("Recording manually stopped: Transcribing...");
		}
		
		StartTranscription();
		return;
	}
	
	// Start recording with Whisper
	AnsiString HomeDir = ExtractFilePath(ExtractFileDir(Application->ExeName));
	RecordedAudioPath = HomeDir + AnsiString("..\\..\\speech\\temp_recording.wav");
	
	// Ensure speech directory exists
	AnsiString speechDir = HomeDir + AnsiString("..\\..\\speech");
	CreateDirectoryA(speechDir.c_str(), NULL);
	
	// Set callback for auto-stop
	if (AudioRecorder)
	{
		AudioRecorder->SetAutoStopCallback((void*)this);
	}
	
	if (AudioRecorder && AudioRecorder->StartRecording(RecordedAudioPath, true))
	{
		IsRecordingVoice = true;
		if (Memo1)
		{
			Memo1->Lines->Add("Recording...");
		}
		printf("WHISPER: Audio recording started with auto-stop: %s\n", RecordedAudioPath.c_str());
	}
	else
	{
		if (Memo1)
		{
			Memo1->Lines->Add("Error: Cannot start audio recording");
		}
		printf("WHISPER ERROR: Failed to start audio recording\n");
		ShowMessage("Cannot start audio recording. Please check if microphone is connected.");
	}
}
//---------------------------------------------------------------------------

