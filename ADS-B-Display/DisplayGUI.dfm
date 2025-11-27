object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'ADS-B Viewer'
  ClientHeight = 652
  ClientWidth = 1212
  Color = clBtnFace
  Constraints.MinHeight = 740
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -10
  Font.Name = 'Tahoma'
  Font.Style = [fsBold]
  Menu = MainMenu1
  Position = poScreenCenter
  WindowState = wsMaximized
  OnMouseWheel = FormMouseWheel
  TextHeight = 12
  object Label16: TLabel
    Left = 13
    Top = 67
    Width = 31
    Height = 13
    Caption = 'ICAO:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'Tahoma'
    Font.Style = [fsBold]
    ParentFont = False
  end
  object Label17: TLabel
    Left = 48
    Top = 67
    Width = 46
    Height = 13
    Caption = 'No Hook'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'Tahoma'
    Font.Style = [fsBold]
    ParentFont = False
  end
  object RightPanel: TPanel
    Left = 962
    Top = 0
    Width = 250
    Height = 652
    Align = alRight
    Color = clWindow
    Constraints.MaxWidth = 250
    TabOrder = 0
    ExplicitLeft = 946
    ExplicitHeight = 603
    object Panel1: TPanel
      Left = 1
      Top = 504
      Width = 248
      Height = 147
      Align = alBottom
      TabOrder = 0
      ExplicitTop = 455
      object Label12: TLabel
        Left = 8
        Top = 8
        Width = 55
        Height = 12
        Caption = 'CPA TIME:'
      end
      object Label19: TLabel
        Left = 8
        Top = 26
        Width = 82
        Height = 12
        Caption = 'CPA DISTANCE:'
      end
      object CpaTimeValue: TLabel
        Left = 69
        Top = 8
        Width = 28
        Height = 12
        Caption = 'NONE'
      end
      object CpaDistanceValue: TLabel
        Left = 96
        Top = 26
        Width = 25
        Height = 12
        Caption = 'None'
      end
      object ZoomIn: TButton
        Left = 5
        Top = 110
        Width = 65
        Height = 18
        Caption = 'Zoom In'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
        TabOrder = 0
        OnClick = ZoomInClick
      end
      object ZoomOut: TButton
        Left = 160
        Top = 110
        Width = 65
        Height = 18
        Caption = 'Zoom Out'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
        TabOrder = 1
        OnClick = ZoomOutClick
      end
      object TimeToGoTrackBar: TTrackBar
        Left = 5
        Top = 71
        Width = 228
        Height = 33
        Max = 1800
        Min = 1
        Frequency = 100
        Position = 1
        TabOrder = 2
        OnChange = TimeToGoTrackBarChange
      end
      object TimeToGoCheckBox: TCheckBox
        Left = 8
        Top = 38
        Width = 105
        Height = 27
        Caption = 'Time-To-Go'
        Checked = True
        State = cbChecked
        TabOrder = 3
      end
      object TimeToGoText: TStaticText
        Left = 119
        Top = 43
        Width = 76
        Height = 16
        Caption = '00:00:00:000'
        TabOrder = 4
      end
    end
    object Panel3: TPanel
      Left = 1
      Top = 1
      Width = 248
      Height = 576
      Align = alTop
      Constraints.MinHeight = 70
      TabOrder = 1
      object Label15: TLabel
        Left = 72
        Top = 422
        Width = 98
        Height = 13
        Caption = 'Areas Of Interest'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
      end
      object DrawMap: TCheckBox
        Left = 5
        Top = 278
        Width = 96
        Height = 17
        Caption = 'Display Map'
        Checked = True
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
        State = cbChecked
        TabOrder = 0
      end
      object PurgeStale: TCheckBox
        Left = 5
        Top = 301
        Width = 96
        Height = 19
        Caption = 'Purge Stale'
        Checked = True
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
        State = cbChecked
        TabOrder = 1
      end
      object CSpinStaleTime: TCSpinEdit
        Left = 107
        Top = 280
        Width = 62
        Height = 21
        Increment = 5
        MaxValue = 1000
        MinValue = 5
        TabOrder = 2
        Value = 90
      end
      object PurgeButton: TButton
        Left = 186
        Top = 283
        Width = 51
        Height = 16
        Caption = 'Purge'
        TabOrder = 3
        OnClick = PurgeButtonClick
      end
      object AreaListView: TListView
        Left = 5
        Top = 441
        Width = 235
        Height = 97
        Columns = <
          item
            Caption = 'Area'
            Width = 170
          end
          item
            Caption = 'Color'
            Width = 40
          end>
        ReadOnly = True
        RowSelect = True
        TabOrder = 4
        ViewStyle = vsReport
        OnCustomDrawItem = AreaListViewCustomDrawItem
        OnSelectItem = AreaListViewSelectItem
      end
      object Insert: TButton
        Left = 5
        Top = 544
        Width = 36
        Height = 18
        Caption = 'Insert'
        TabOrder = 5
        OnClick = InsertClick
      end
      object Delete: TButton
        Left = 158
        Top = 544
        Width = 44
        Height = 18
        Caption = 'Delete'
        Enabled = False
        TabOrder = 6
        OnClick = DeleteClick
      end
      object Complete: TButton
        Left = 45
        Top = 544
        Width = 57
        Height = 18
        Caption = 'Complete'
        Enabled = False
        TabOrder = 7
        OnClick = CompleteClick
      end
      object Cancel: TButton
        Left = 107
        Top = 544
        Width = 41
        Height = 18
        Caption = 'Cancel'
        Enabled = False
        TabOrder = 8
        OnClick = CancelClick
      end
      object RawConnectButton: TButton
        Left = 5
        Top = 326
        Width = 104
        Height = 18
        Caption = 'Raw Connect'
        TabOrder = 9
        OnClick = RawConnectButtonClick
      end
      object RawIpAddress: TEdit
        Left = 115
        Top = 326
        Width = 110
        Height = 20
        TabOrder = 10
        Text = '127.0.0.1'
      end
      object CycleImages: TCheckBox
        Left = 107
        Top = 301
        Width = 96
        Height = 19
        Caption = 'Cycle Images'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'Tahoma'
        Font.Style = [fsBold]
        ParentFont = False
        TabOrder = 11
        OnClick = CycleImagesClick
      end
      object Panel4: TPanel
        Left = 5
        Top = 76
        Width = 244
        Height = 198
        BevelInner = bvLowered
        BevelKind = bkFlat
        TabOrder = 12
        object CLatLabel: TLabel
          Left = 66
          Top = 57
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object CLonLabel: TLabel
          Left = 66
          Top = 76
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
          Transparent = False
        end
        object SpdLabel: TLabel
          Left = 66
          Top = 92
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object HdgLabel: TLabel
          Left = 66
          Top = 111
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object AltLabel: TLabel
          Left = 66
          Top = 130
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object MsgCntLabel: TLabel
          Left = 66
          Top = 149
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object TrkLastUpdateTimeLabel: TLabel
          Left = 120
          Top = 168
          Width = 72
          Height = 13
          Caption = '00:00:00:000'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label14: TLabel
          Left = 5
          Top = 168
          Width = 109
          Height = 13
          Caption = 'LAST UPDATE TIME:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label13: TLabel
          Left = 5
          Top = 150
          Width = 52
          Height = 13
          Caption = 'MSG CNT:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label10: TLabel
          Left = 5
          Top = 131
          Width = 24
          Height = 13
          Caption = 'ALT:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label9: TLabel
          Left = 5
          Top = 112
          Width = 27
          Height = 13
          Caption = 'HDG:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label8: TLabel
          Left = 5
          Top = 93
          Width = 25
          Height = 13
          Caption = 'SPD:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label7: TLabel
          Left = 5
          Top = 76
          Width = 24
          Height = 13
          Caption = 'LON:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label6: TLabel
          Left = 5
          Top = 57
          Width = 24
          Height = 13
          Caption = 'LAT:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label18: TLabel
          Left = 5
          Top = 38
          Width = 55
          Height = 13
          Caption = 'FLIGHT #:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object FlightNumLabel: TLabel
          Left = 66
          Top = 38
          Width = 21
          Height = 13
          Caption = 'N/A'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object ICAOLabel: TLabel
          Left = 66
          Top = 22
          Width = 46
          Height = 13
          Caption = 'No Hook'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label5: TLabel
          Left = 5
          Top = 19
          Width = 31
          Height = 13
          Caption = 'ICAO:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label4: TLabel
          Left = 75
          Top = 0
          Width = 85
          Height = 16
          Caption = 'Close Control'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -13
          Font.Name = 'Tahoma'
          Font.Style = [fsBold, fsUnderline]
          ParentFont = False
        end
        object Label20: TLabel
          Left = 5
          Top = 181
          Width = 40
          Height = 13
          Caption = 'ROUTE:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object RouteLabel: TLabel
          Left = 51
          Top = 181
          Width = 55
          Height = 13
          Caption = 'UNKNOWN'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
      end
      object Panel5: TPanel
        Left = 5
        Top = 2
        Width = 244
        Height = 77
        BevelInner = bvLowered
        BevelKind = bkFlat
        TabOrder = 13
        object Lon: TLabel
          Left = 31
          Top = 40
          Width = 7
          Height = 13
          Caption = '0'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label3: TLabel
          Left = 5
          Top = 40
          Width = 23
          Height = 13
          Caption = 'Lon:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Lat: TLabel
          Left = 31
          Top = 21
          Width = 7
          Height = 13
          Caption = '0'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label2: TLabel
          Left = 5
          Top = 21
          Width = 21
          Height = 13
          Caption = 'Lat:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object SystemTimeLabel: TLabel
          Left = 5
          Top = 4
          Width = 74
          Height = 13
          Caption = 'System Time'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object ViewableAircraftCountLabel: TLabel
          Left = 157
          Top = 59
          Width = 7
          Height = 12
          Caption = '0'
        end
        object AircraftCountLabel: TLabel
          Left = 66
          Top = 58
          Width = 7
          Height = 13
          Caption = '0'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label11: TLabel
          Left = 101
          Top = 58
          Width = 53
          Height = 13
          Caption = 'Viewable:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object Label1: TLabel
          Left = 5
          Top = 58
          Width = 58
          Height = 13
          Caption = '# Aircraft:'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
        end
        object SystemTime: TStaticText
          Left = 85
          Top = 4
          Width = 76
          Height = 17
          Caption = '00:00:00:000'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clWindowText
          Font.Height = -11
          Font.Name = 'Tahoma'
          Font.Style = [fsBold]
          ParentFont = False
          TabOrder = 0
        end
      end
      object RawPlaybackButton: TButton
        Left = 133
        Top = 350
        Width = 96
        Height = 17
        Caption = 'Raw Playback'
        TabOrder = 14
        OnClick = RawPlaybackButtonClick
      end
      object RawRecordButton: TButton
        Left = 5
        Top = 350
        Width = 104
        Height = 17
        Caption = 'Raw Record'
        TabOrder = 15
        OnClick = RawRecordButtonClick
      end
      object SBSConnectButton: TButton
        Left = 5
        Top = 373
        Width = 104
        Height = 20
        Caption = 'SBS Connect'
        TabOrder = 16
        OnClick = SBSConnectButtonClick
      end
      object SBSIpAddress: TEdit
        Left = 115
        Top = 373
        Width = 117
        Height = 20
        TabOrder = 17
        Text = 'data.adsbhub.org'
      end
      object SBSRecordButton: TButton
        Left = 5
        Top = 399
        Width = 104
        Height = 17
        Caption = 'SBS Record'
        TabOrder = 18
        OnClick = SBSRecordButtonClick
      end
      object SBSPlaybackButton: TButton
        Left = 115
        Top = 399
        Width = 96
        Height = 17
        Caption = 'SBS Playback'
        TabOrder = 19
        OnClick = SBSPlaybackButtonClick
      end
      object Memo1: TMemo
        Left = 0
        Top = 424
        Width = 249
        Height = 144
        Lines.Strings = (
          '')
        ReadOnly = True
        TabOrder = 20
        Visible = False
      end
    end
    object Panel2: TPanel
      Left = 1
      Top = 577
      Width = 248
      Height = 9
      Align = alClient
      TabOrder = 2
      object MapComboBox: TComboBox
        Left = 5
        Top = 6
        Width = 100
        Height = 20
        Style = csDropDownList
        TabOrder = 0
        OnChange = MapComboBoxChange
        Items.Strings = (
          'Google Maps'
          'VFR'
          'IFS Low'
          'IFS High')
      end
      object BigQueryCheckBox: TCheckBox
        Left = 5
        Top = 32
        Width = 124
        Height = 17
        Caption = 'BigQuery Upload'
        TabOrder = 1
        OnClick = BigQueryCheckBoxClick
      end
    end
  end
  object ObjectDisplay: TOpenGLPanel
    Left = 0
    Top = 0
    Width = 962
    Height = 652
    Cursor = crCross
    Align = alClient
    TabOrder = 1
    PFDFlags = [f_PFD_DRAW_TO_WINDOW, f_PFD_SUPPORT_OPENGL, f_PFD_DOUBLEBUFFER]
    Font3D_Type.Charset = ANSI_CHARSET
    Font3D_Type.Color = clWindowText
    Font3D_Type.Height = -9
    Font3D_Type.Name = 'Arial'
    Font3D_Type.Style = []
    Font3D_Extrustion = 0.100000001490116100
    Font2D_Enabled = True
    Font2D_Type.Charset = ANSI_CHARSET
    Font2D_Type.Color = clWindowText
    Font2D_Type.Height = -27
    Font2D_Type.Name = 'Arial'
    Font2D_Type.Style = [fsBold]
    OnMouseDown = ObjectDisplayMouseDown
    OnMouseMove = ObjectDisplayMouseMove
    OnMouseUp = ObjectDisplayMouseUp
    OnInit = ObjectDisplayInit
    OnResize = ObjectDisplayResize
    OnPaint = ObjectDisplayPaint
    ExplicitWidth = 946
    ExplicitHeight = 603
  end
  object MainMenu1: TMainMenu
    Left = 24
    object File1: TMenuItem
      Caption = 'File'
      object UseSBSRemote: TMenuItem
        Caption = 'ADS-B Hub'
        OnClick = UseSBSRemoteClick
      end
      object UseSBSLocal: TMenuItem
        Caption = 'ADS-B Local'
        OnClick = UseSBSLocalClick
      end
      object LIsten: TMenuItem
        Caption = 'Listen'
        OnClick = LIstenClick
      end
      object LoadARTCCBoundaries1: TMenuItem
        Caption = 'Load ARTCC Boundaries'
        OnClick = LoadARTCCBoundaries1Click
      end
      object Exit1: TMenuItem
        Caption = 'Exit'
        OnClick = Exit1Click
      end
    end
  end
  object Timer1: TTimer
    Interval = 500
    OnTimer = Timer1Timer
    Left = 88
  end
  object Timer2: TTimer
    Interval = 5000
    OnTimer = Timer2Timer
    Left = 136
  end
  object IdTCPClientRaw: TIdTCPClient
    OnDisconnected = IdTCPClientRawDisconnected
    OnConnected = IdTCPClientRawConnected
    ConnectTimeout = 0
    Port = 0
    ReadTimeout = -1
    ReuseSocket = rsTrue
    Left = 208
  end
  object RecordRawSaveDialog: TSaveDialog
    DefaultExt = 'raw'
    Filter = 'raw|*.raw'
    Left = 328
  end
  object PlaybackRawDialog: TOpenDialog
    DefaultExt = 'raw'
    Filter = 'raw|*.raw'
    Left = 448
  end
  object IdTCPClientSBS: TIdTCPClient
    OnDisconnected = IdTCPClientSBSDisconnected
    OnConnected = IdTCPClientSBSConnected
    ConnectTimeout = 0
    Port = 0
    ReadTimeout = -1
    ReuseSocket = rsTrue
    Left = 552
  end
  object RecordSBSSaveDialog: TSaveDialog
    DefaultExt = 'sbs'
    Filter = 'sbs|*.sbs'
    Left = 664
  end
  object PlaybackSBSDialog: TOpenDialog
    DefaultExt = 'sbs'
    Filter = 'sbs|*.sbs'
    Left = 784
  end
  object NetHTTPClientRoute: TNetHTTPClient
    UserAgent = 'Embarcadero URI Client/1.0'
    Left = 40
    Top = 112
  end
  object SpVoice1: TSpVoice
    AutoConnect = False
    ConnectKind = ckRunningOrNew
    Left = 168
    Top = 120
  end
  object SpSharedRecoContext1: TSpSharedRecoContext
    AutoConnect = False
    ConnectKind = ckRunningOrNew
    OnRecognition = SpSharedRecoContext1Recognition
    Left = 280
    Top = 144
  end
end
