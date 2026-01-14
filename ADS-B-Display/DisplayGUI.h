//---------------------------------------------------------------------------

#ifndef DisplayGUIH
#define DisplayGUIH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include <Controls.hpp>
#include <StdCtrls.hpp>
#include <Forms.hpp>
#include "Components\OpenGLv0.5BDS2006\Component\OpenGLPanel.h"
#include <ComCtrls.hpp>
#include <ExtCtrls.hpp>
#include <Menus.hpp>
#include <IdBaseComponent.hpp>
#include <IdComponent.hpp>
#include <Graphics.hpp>
#include "FilesystemStorage.h"
#include "KeyholeConnection.h"
#include "GoogleLayer.h"
#include "FlatEarthView.h"
#include "ght_hash_table.h"
#include "TriangulatPoly.h"
#include <Dialogs.hpp>
#include <IdTCPClient.hpp>
#include <IdTCPConnection.hpp>
#include "cspin.h"
#include <System.Net.HttpClient.hpp>
#include <System.Net.HttpClientComponent.hpp>
#include <System.Net.URLClient.hpp>
#include <windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#include "SpeechLib_OCX.h"
#include <Vcl.OleServer.hpp>

typedef float T_GL_Color[4];


typedef struct
{
 bool Valid_CC;
 bool Valid_CPA;
 uint32_t ICAO_CC;
 uint32_t ICAO_CPA;
}TTrackHook;

typedef struct
{
 double lat;
 double lon;
 double hae;
}TPolyLine;


#define MAX_AREA_POINTS 500
typedef struct
{
 AnsiString  Name;
 TColor      Color;
 DWORD       NumPoints;
 pfVec3      Points[MAX_AREA_POINTS];
 pfVec3      PointsAdj[MAX_AREA_POINTS];
 TTriangles *Triangles;
 bool        Selected;
}TArea;
//---------------------------------------------------------------------------
class  TTCPClientRawHandleThread : public TThread
{
private:
	AnsiString StringMsgBuffer;
	void __fastcall HandleInput(void);
	void __fastcall StopPlayback(void);
	void __fastcall StopTCPClient(void);
protected:
	void __fastcall Execute(void);
public:
	 bool UseFileInsteadOfNetwork;
	 bool First;
	 __int64 LastTime;
	__fastcall TTCPClientRawHandleThread(bool value);
	~TTCPClientRawHandleThread();
};
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class  TTCPClientSBSHandleThread : public TThread
{
private:
	AnsiString StringMsgBuffer;
	void __fastcall HandleInput(void);
	void __fastcall StopPlayback(void);
	void __fastcall StopTCPClient(void);
protected:
	void __fastcall Execute(void);
public:
	 bool UseFileInsteadOfNetwork;
	 bool First;
	 __int64 LastTime;
	__fastcall TTCPClientSBSHandleThread(bool value);
	~TTCPClientSBSHandleThread();
};
//---------------------------------------------------------------------------
// Audio Recording Helper Class
//---------------------------------------------------------------------------
class TAudioRecorder
{
private:
	::HWAVEIN hWaveIn;
	::WAVEFORMATEX wfx;
	::WAVEHDR* waveHeaders;
	int numBuffers;
	bool isRecording;
	AnsiString outputFilePath;
	
	// Silence detection
	double silenceThreshold;  // RMS threshold for silence (0.0 to 1.0)
	int silenceDurationMs;    // Duration of silence before auto-stop (milliseconds)
	int currentSilenceMs;     // Current silence duration
	DWORD lastSoundTime;      // Time when last sound was detected
	bool autoStopEnabled;     // Enable automatic stop on silence
	void* autoStopCallback;   // Callback function pointer (TForm1*)
	
public:
	__fastcall TAudioRecorder();
	__fastcall ~TAudioRecorder();
	bool StartRecording(AnsiString filePath, bool enableAutoStop = true);
	void StopRecording();
	bool IsRecording() { return isRecording; }
	AnsiString GetOutputPath() { return outputFilePath; }
	void SetAutoStopCallback(void* callback) { autoStopCallback = callback; }
	
private:
	static void CALLBACK WaveInProc(::HWAVEIN hWaveIn, UINT uMsg, ::DWORD_PTR dwInstance, ::DWORD_PTR dwParam1, ::DWORD_PTR dwParam2);
	void ProcessBuffer(::WAVEHDR* pwh);
	double CalculateRMS(short* samples, int numSamples);
	void CheckSilenceAndAutoStop();
};
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class TSpeechTranscribeThread : public TThread
{
private:
	AnsiString AudioFilePath;
	AnsiString PythonPath;
	AnsiString TranscribeScriptPath;
	AnsiString TranscribedText;
	bool TranscriptionSuccess;
	void __fastcall HandleTranscriptionResult(void);
protected:
	void __fastcall Execute(void);
public:
	__fastcall TSpeechTranscribeThread(AnsiString audioFile, AnsiString pythonPath, AnsiString scriptPath);
	~TSpeechTranscribeThread();
};
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// Message ID for auto-stop recording
#define WM_AUTO_STOP_RECORDING (WM_USER + 1)

class TForm1 : public TForm
{
private:
	void __fastcall WMAutoStopRecording(TMessage &Message);
	void StartTranscription(); // Helper function to start transcription
	
__published:	// IDE-managed Components
	TMainMenu *MainMenu1;
	TPanel *RightPanel;
	TMenuItem *File1;
	TMenuItem *Exit1;
	TTimer *Timer1;
	TOpenGLPanel *ObjectDisplay;
	TPanel *Panel1;
	TPanel *Panel3;
	TButton *ZoomIn;
	TButton *ZoomOut;
	TCheckBox *DrawMap;
	TCheckBox *PurgeStale;
	TTimer *Timer2;
	TCSpinEdit *CSpinStaleTime;
	TButton *PurgeButton;
	TListView *AreaListView;
	TButton *Insert;
	TButton *Delete;
	TButton *JFKButton;
	TButton *Complete;
	TButton *Cancel;
	TButton *RawConnectButton;
	TLabel *Label16;
	TLabel *Label17;
	TEdit *RawIpAddress;
	TIdTCPClient *IdTCPClientRaw;
	TSaveDialog *RecordRawSaveDialog;
	TOpenDialog *PlaybackRawDialog;
	TCheckBox *CycleImages;
	TPanel *Panel4;
	TLabel *CLatLabel;
	TLabel *CLonLabel;
	TLabel *SpdLabel;
	TLabel *HdgLabel;
	TLabel *AltLabel;
	TLabel *MsgCntLabel;
	TLabel *TrkLastUpdateTimeLabel;
	TLabel *Label14;
	TLabel *Label13;
	TLabel *Label10;
	TLabel *Label9;
	TLabel *Label8;
	TLabel *Label7;
	TLabel *Label6;
	TLabel *Label18;
	TLabel *FlightNumLabel;
	TLabel *ICAOLabel;
	TLabel *Label5;
	TLabel *Label4;
	TPanel *Panel5;
	TLabel *Lon;
	TLabel *Label3;
	TLabel *Lat;
	TLabel *Label2;
	TStaticText *SystemTime;
	TLabel *SystemTimeLabel;
	TLabel *ViewableAircraftCountLabel;
	TLabel *AircraftCountLabel;
	TLabel *Label11;
	TLabel *Label1;
	TButton *RawPlaybackButton;
	TButton *RawRecordButton;
	TIdTCPClient *IdTCPClientSBS;
	TButton *SBSConnectButton;
	TEdit *SBSIpAddress;
	TButton *SBSRecordButton;
	TButton *SBSPlaybackButton;
	TSaveDialog *RecordSBSSaveDialog;
	TOpenDialog *PlaybackSBSDialog;
	TTrackBar *TimeToGoTrackBar;
	TCheckBox *TimeToGoCheckBox;
	TStaticText *TimeToGoText;
	TLabel *Label12;
	TLabel *Label19;
	TLabel *CpaTimeValue;
	TLabel *CpaDistanceValue;
	TPanel *Panel2;
	TComboBox *MapComboBox;
	TCheckBox *BigQueryCheckBox;
	TMenuItem *UseSBSLocal;
	TMenuItem *UseSBSRemote;
	TMenuItem *LoadARTCCBoundaries1;
	TNetHTTPClient *NetHTTPClientRoute;
	TLabel *Label20;
	TLabel *RouteLabel;
	TSpVoice *SpVoice1;
	TSpSharedRecoContext *SpSharedRecoContext1;
	TMemo *Memo1;
	TMenuItem *LIsten;
	void __fastcall ObjectDisplayInit(TObject *Sender);
	void __fastcall ObjectDisplayResize(TObject *Sender);
	void __fastcall ObjectDisplayPaint(TObject *Sender);
	void __fastcall Timer1Timer(TObject *Sender);
	void __fastcall ResetXYOffset(void);
	void __fastcall ObjectDisplayMouseDown(TObject *Sender, TMouseButton Button,
		  TShiftState Shift, int X, int Y);
	void __fastcall ObjectDisplayMouseMove(TObject *Sender, TShiftState Shift,
		  int X, int Y);
	void __fastcall AddPoint(int X, int Y);	  
	void __fastcall ObjectDisplayMouseUp(TObject *Sender, TMouseButton Button,
          TShiftState Shift, int X, int Y);
	void __fastcall Exit1Click(TObject *Sender);
	void __fastcall ZoomInClick(TObject *Sender);
	void __fastcall ZoomOutClick(TObject *Sender);
	void __fastcall Timer2Timer(TObject *Sender);
	void __fastcall PurgeButtonClick(TObject *Sender);
	void __fastcall InsertClick(TObject *Sender);
	void __fastcall CancelClick(TObject *Sender);
	void __fastcall CompleteClick(TObject *Sender);
	void __fastcall AreaListViewSelectItem(TObject *Sender, TListItem *Item,
          bool Selected);
	void __fastcall DeleteClick(TObject *Sender);
	void __fastcall JFKButtonClick(TObject *Sender);
	void __fastcall AreaListViewCustomDrawItem(TCustomListView *Sender,
          TListItem *Item, TCustomDrawState State, bool &DefaultDraw);
	void __fastcall FormMouseWheel(TObject *Sender, TShiftState Shift,
          int WheelDelta, TPoint &MousePos, bool &Handled);
	void __fastcall RawConnectButtonClick(TObject *Sender);
	void __fastcall IdTCPClientRawConnected(TObject *Sender);
	void __fastcall RawRecordButtonClick(TObject *Sender);
	void __fastcall RawPlaybackButtonClick(TObject *Sender);
	void __fastcall IdTCPClientRawDisconnected(TObject *Sender);
	void __fastcall CycleImagesClick(TObject *Sender);
	void __fastcall SBSConnectButtonClick(TObject *Sender);
	void __fastcall SBSRecordButtonClick(TObject *Sender);
	void __fastcall SBSPlaybackButtonClick(TObject *Sender);
	void __fastcall IdTCPClientSBSConnected(TObject *Sender);
	void __fastcall IdTCPClientSBSDisconnected(TObject *Sender);
	void __fastcall TimeToGoTrackBarChange(TObject *Sender);
	void __fastcall MapComboBoxChange(TObject *Sender);
	void __fastcall BigQueryCheckBoxClick(TObject *Sender);
	void __fastcall UseSBSRemoteClick(TObject *Sender);
	void __fastcall UseSBSLocalClick(TObject *Sender);
	void __fastcall LoadARTCCBoundaries1Click(TObject *Sender);
	void __fastcall SpSharedRecoContext1Recognition(TObject *Sender, long StreamNumber,
          Variant StreamPosition, SpeechRecognitionType RecognitionType,
          ISpeechRecoResult *Result);
	void __fastcall LIstenClick(TObject *Sender);
	void __fastcall ProcessVoiceCommand(AnsiString command);

private:	// User declarations


public:		// User declarations
	__fastcall TForm1(TComponent* Owner);
	__fastcall ~TForm1();
	void __fastcall LatLon2XY(double lat,double lon, double &x, double &y);
	int __fastcall  XY2LatLon2(int x, int y,double &lat,double &lon );
	void __fastcall HookTrack(int X, int Y,bool CPA_Hook);
	void __fastcall DrawObjects(void);
	void __fastcall DeleteAllAreas(void);
	void __fastcall Purge(void);
	int __fastcall CountFlightsInArea(TArea *Area);
	void __fastcall UpdateAreaFlightCounts(void);
	int __fastcall CountFlightsInRadius(double centerLat, double centerLon, double radiusMiles);
	double __fastcall CalculateDistanceMiles(double lat1, double lon1, double lat2, double lon2);
	void __fastcall SendCotMessage(AnsiString IP_address, unsigned short Port,char *Buffer,DWORD Length);
	void __fastcall RegisterWithCoTRouter(void);
    void __fastcall SetMapCenter(double &x, double &y);
    void __fastcall LoadMap(int Type);
    void __fastcall CreateBigQueryCSV(void);
    void __fastcall CloseBigQueryCSV(void);
    bool __fastcall LoadARTCCBoundaries(AnsiString FileName);

    ISpeechRecoGrammar         *SRGrammar;
	int                        MouseDownX,MouseDownY;
	bool                       MouseDown;
	TTrackHook                 TrackHook;
	Vector3d                   Map_v[4],Map_p[4];
	Vector2d                   Map_w[2];
	double                     Mw1,Mw2,Mh1,Mh2,xf,yf;
	KeyholeConnection	      *g_Keyhole;
	FilesystemStorage	      *g_Storage;
	MasterLayer	      	      *g_MasterLayer;
	TileManager		          *g_GETileManager;
	EarthView		          *g_EarthView;
	double                     MapCenterLat,MapCenterLon;
	int			               g_MouseLeftDownX;
	int			               g_MouseLeftDownY;
	int			               g_MouseDownMask ;
	bool                       LoadMapFromInternet;
	TList                     *Areas;
	TArea                     *AreaTemp;
	ght_hash_table_t          *HashTable;
	TTCPClientRawHandleThread *TCPClientRawHandleThread;
    TTCPClientSBSHandleThread *TCPClientSBSHandleThread;
	TStreamWriter              *RecordRawStream;
	TStreamReader              *PlayBackRawStream;
    TStreamWriter              *RecordSBSStream;
	TStreamReader              *PlayBackSBSStream;
	TStreamWriter              *BigQueryCSV;
    AnsiString                 BigQueryCSVFileName;
	unsigned int               BigQueryRowCount;
	unsigned int               BigQueryFileCount;
    AnsiString                 BigQueryPythonScript;
	AnsiString                 BigQueryPath;
    AnsiString                 BigQueryLogFileName;
	int                        NumSpriteImages;
	int                        CurrentSpriteImage;
    AnsiString                 AircraftDBPathFileName;
    AnsiString                 ARTCCBoundaryDataPathFileName;
    AnsiString                 SpeechPythonPath;
    AnsiString                 SpeechTranscribeScriptPath;
    TSpeechTranscribeThread   *SpeechTranscribeThread;
    bool                       IsRecordingVoice;
    AnsiString                 RecordedAudioPath;
    TAudioRecorder             *AudioRecorder;
    
    // Airport radius tracking
    bool                       TrackingAirport;
    AnsiString                 AirportName;
    double                     AirportLat;
    double                     AirportLon;
    double                     AirportRadiusMiles;
    TListItem                 *AirportListItem;
    
    BEGIN_MESSAGE_MAP
        MESSAGE_HANDLER(WM_AUTO_STOP_RECORDING, TMessage, WMAutoStopRecording)
    END_MESSAGE_MAP(TForm)
};
//---------------------------------------------------------------------------
extern PACKAGE TForm1 *Form1;
//---------------------------------------------------------------------------


#endif
