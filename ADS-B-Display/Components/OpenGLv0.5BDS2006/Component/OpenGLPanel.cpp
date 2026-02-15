//---------------------------------------------------------------------------
//|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//| TOpenGLPanel.cpp Version 0.5BCB5 ALPHA
//|
//| This is a implementation of a OpenGL Component for C++ Builder Version 5.0.
//|
//| This code was created by Daniel Plakosh. There are no restrictions on the
//| use or distribution of this code, except that you may not restrict its use
//| or distribution. Any project which is created from this code, or any
//| significant modification of this code, may be distributed in any way you
//| choose; you may even sell it or its source code. However, any limitations
//| placed on the distribution or use of such a project or modification may not
//| in any way limit the distribution or use of this original code.
//|
//| The developer makes no representations about the suitability of this
//| software for any purpose. It is provided "as is" without express or implied
//| warranty.
//|
//| Contact Daniel Plakosh at <dplakosh@cobweb.net> or
//| <dplakosh@sei.example.org> to request assistance or to report a problem with
//| this software. Criticism, suggestions and comments are also greatly
//| appreciated. Remember to always include the TOpenGLPanel version and system
//| information in your messages.
//|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//---------------------------------------------------------------------------
#include <vcl.h>

#pragma hdrstop
#include <float.h>
#include <stdlib.h>
#include <Printers.hpp>
#include "OpenGLPanel.h"
#pragma package(smart_init)
//---------------------------------------------------------------------------
// ValidCtrCheck is used to assure that the components created do not have
// any pure virtual functions.
//

static inline void ValidCtrCheck(TOpenGLPanel *)
{
	new TOpenGLPanel(NULL);
}
//---------------------------------------------------------------------------
__fastcall TOpenGLPanel::TOpenGLPanel(TComponent* Owner)
	: TCustomPanel(Owner)
{
 //_control87(MCW_EM, MCW_EM);
 FdwFlags.Clear();
 FdwFlags << f_PFD_DRAW_TO_WINDOW << f_PFD_SUPPORT_OPENGL <<f_PFD_DOUBLEBUFFER;
 FiPixelType=f_PFD_TYPE_RGBA;
 SavedPixelType=PFD_TYPE_RGBA;
 FcColorBits=32;
 FcDepthBits=32;
 FSwapBuffers=Auto;

 FOpenGLPrintScale=pglProportional;
 FOpenGLPixelsPerInch=96;

 Font3DEnabled=false;
 Font3DType= new TFont;
 Font3DType->Name="Arial";
 Font3DType->Style<< fsBold;
 Font3DType->Height=-9;
 Font3DType->Charset=ANSI_CHARSET;
 Font3DFirstGylph=32;
 Font3DNumGylph=96;
 Font3DMaxDeviation=0.0;
 Font3DExtrusion=0.1;
 Font3DFormat=f_WGL_FONT_POLYGONS;
 SavedFont3DFormat=WGL_FONT_POLYGONS;
 Font3DDefault=NULL;

 Font2DEnabled=false;
 Font2DType= new TFont;
 Font2DType->Name="Arial";
 Font2DType->Style<< fsBold;
 Font2DType->Height=-9;
 Font2DType->Charset=ANSI_CHARSET;
 Font2DFirstGylph=32;
 Font2DNumGylph=96;
 Font2DDefault=NULL;
}
//---------------------------------------------------------------------------
namespace Openglpanel
{
	void __fastcall PACKAGE Register()
	{
		TComponentClass classes[1] = {__classid(TOpenGLPanel)};
		RegisterComponents(L"OpenGL_DP", classes, 0);
	}
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private Methods (non-Property)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//---------------------------------------------------------------------------
 bool __fastcall TOpenGLPanel::CreateGLContext(void)
 {
  int PixelFormatIndex;
  PIXELFORMATDESCRIPTOR pfd=
    {
     sizeof(PIXELFORMATDESCRIPTOR), // size
     1,                             // version 1
     0,                             // Flags
     SavedPixelType,                // Color Mode
     FcColorBits,                   // Number of color bits
     0,0,0,0,0,0,                   // RGB color bits with shift (not used)
     0,0,                           // Alpha bits with shift     (not used)
     0,                             // Number of accumulator bits
     0,0,0,0,                       // RGBA accumulator bits
     FcDepthBits,                   // Number of bits for Z-Buffer Depth
     0,                             // Stencil buffer depth
     0,                             // Auxiliary buffer depth
     PFD_MAIN_PLANE,                // Layer type
     0,                             // Reserved
     0,                             // Layer mask
     0,                             // Visible mask
     0                              // Damage mask
    };

  if (FdwFlags.Contains(f_PFD_DRAW_TO_WINDOW))
                 pfd.dwFlags|=PFD_DRAW_TO_WINDOW;
  if (FdwFlags.Contains(f_PFD_DRAW_TO_BITMAP))
                 pfd.dwFlags|= PFD_DRAW_TO_BITMAP;
  if (FdwFlags.Contains(f_PFD_SUPPORT_GDI))
                 pfd.dwFlags|= PFD_SUPPORT_GDI;
  else           pfd.dwFlags|= PFD_DOUBLEBUFFER;
  if (FdwFlags.Contains(f_PFD_SUPPORT_OPENGL))
                 pfd.dwFlags|= PFD_SUPPORT_OPENGL;
  if (FdwFlags.Contains(f_PFD_GENERIC_ACCELERATED))
                 pfd.dwFlags|= PFD_GENERIC_ACCELERATED;
  if (FdwFlags.Contains(f_PFD_GENERIC_FORMAT))
                 pfd.dwFlags|= PFD_GENERIC_FORMAT;
  if (FdwFlags.Contains(f_PFD_NEED_PALETTE))
                 pfd.dwFlags|=PFD_NEED_PALETTE ;
  if (FdwFlags.Contains(f_PFD_NEED_SYSTEM_PALETTE))
                 pfd.dwFlags|= PFD_NEED_SYSTEM_PALETTE;
  if (FdwFlags.Contains(f_PFD_STEREO))
                 pfd.dwFlags|= PFD_STEREO;
  if (FdwFlags.Contains(f_PFD_SWAP_LAYER_BUFFERS))
                 pfd.dwFlags|=PFD_SWAP_LAYER_BUFFERS;

  DisplayDeviceContext=GetDC(this->Handle);
  if ((PixelFormatIndex = ChoosePixelFormat(DisplayDeviceContext,&pfd))==0)
        {
         ShowMessage("ChoosePixelFormat:" + IntToStr((int)GetLastError()));
         ReleaseDC(this->Handle,DisplayDeviceContext);
         return false;
        }

  if (SetPixelFormat(DisplayDeviceContext,PixelFormatIndex,&pfd)!=true)
        {
         ShowMessage("SetPixelFormat:" + IntToStr((int)GetLastError()));
         ReleaseDC(this->Handle,DisplayDeviceContext);
         return false;
        }
 if (pfd.dwFlags & PFD_DOUBLEBUFFER) DoubleBuffer=true;
 else DoubleBuffer=false;

 DescribePixelFormat(DisplayDeviceContext,PixelFormatIndex,
                       sizeof(PIXELFORMATDESCRIPTOR),&pfd);
  if (pfd.dwFlags & PFD_NEED_PALETTE)
        {
         if (ConstructPalette(pfd)==false)
             {
              ReleaseDC(this->Handle,DisplayDeviceContext);
              return false;
             }
        }
  else Palette=NULL;
   if ((GLRenderingContext=wglCreateContext(DisplayDeviceContext))==NULL)
        {
         ShowMessage("wglRenderingContext:" + IntToStr((int)GetLastError()));
         if (Palette)
            {
			 SelectPalette(DisplayDeviceContext,
						   (HPALETTE)GetStockObject(DEFAULT_PALETTE),false);
			 if (DeleteObject(Palette)==false)
                 ShowMessage("DeleteObject: Failed");
            }
         ReleaseDC(this->Handle,DisplayDeviceContext);
         return false;
        }
   return true;
 }
 //---------------------------------------------------------------------------
 void __fastcall TOpenGLPanel::DestroyGLContext(void)
 {
  delete Font3DDefault;
  delete Font2DDefault;
  wglMakeCurrent(NULL,NULL);
  if (GLRenderingContext!=NULL)
        {
         if (wglDeleteContext(GLRenderingContext)==false)
         	ShowMessage("wglDeleteContext:"+IntToStr((int)GetLastError()));
         if ((Palette) && (DeleteObject(Palette)==false))
                     ShowMessage("DeleteObject: Failed");
		 if (ReleaseDC(this->Handle,(HDC)GLRenderingContext)!=0)
           ShowMessage("ReleaseDC: Failed");
        }
  return;
 }
 //---------------------------------------------------------------------------
 bool __fastcall TOpenGLPanel::ConstructPalette(PIXELFORMATDESCRIPTOR &pfd)
 {
  LOGPALETTE *LPalettep;
  WORD NumColors= (WORD)(1 << pfd.cColorBits);

  if (( LPalettep=(LOGPALETTE * )
             malloc(sizeof(LOGPALETTE)+(sizeof(PALETTEENTRY)*NumColors)))==NULL)
       {
        ShowMessage("ConstructPalette: Malloc failure");
        return false;
       }
  LPalettep->palVersion=0x300;
  LPalettep->palNumEntries=NumColors;

  int RedMask   =( 1 << pfd.cRedBits  ) -1;
  int GreenMask =( 1 << pfd.cGreenBits) -1;
  int BlueMask  =( 1 << pfd.cBlueBits ) -1;

  for (int i=0;i<NumColors;i++)
        {
         LPalettep->palPalEntry[i].peRed=(BYTE)
         	((((i >> pfd.cRedShift) & RedMask)     *255)/RedMask);

         LPalettep->palPalEntry[i].peGreen=(BYTE)
         	((((i >> pfd.cGreenShift) & GreenMask) *255)/GreenMask);

         LPalettep->palPalEntry[i].peBlue=(BYTE)
         	((((i >> pfd.cBlueShift) &  BlueMask)  *255)/BlueMask);
        }
  if ((Palette=CreatePalette(LPalettep))==NULL)
        {
         ShowMessage("CreatePalette:"+IntToStr((int)GetLastError()));
         return false;
        }
  free(LPalettep);
  return true;
 }
 //---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Create3DFont(void)
  {
   delete Font3DDefault;
   Font3DDefault=Create3DFont(Font3DType,Font3DFirstGylph,Font3DNumGylph,
                              Font3DMaxDeviation,Font3DExtrusion,
                              SavedFont3DFormat);
  }
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Create2DFont(void)
  {
   delete Font2DDefault;
   Font2DDefault=Create2DFont(Font2DType,Font2DFirstGylph,Font2DNumGylph);
  }
//---------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private Methods (Property)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetPFDPixelType(const TPFDPixelTypes Value)
  {
   if (Value==f_PFD_TYPE_RGBA) SavedPixelType=PFD_TYPE_RGBA;
   else SavedPixelType=PFD_TYPE_COLORINDEX;
   FiPixelType=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetPFDFlags(const TPFDFlagsSet Value)
  {
   TPFDFlagsSet Temp=Value;

   if ((Value.Contains(f_PFD_SUPPORT_GDI)) &&
       (Value.Contains(f_PFD_DOUBLEBUFFER)))
       {
        if (FdwFlags.Contains(f_PFD_SUPPORT_GDI)) Temp >> f_PFD_SUPPORT_GDI;
        else Temp >> f_PFD_DOUBLEBUFFER;
       }
   else if ((!Value.Contains(f_PFD_SUPPORT_GDI)) &&
       (!Value.Contains(f_PFD_DOUBLEBUFFER)))
       {
        if (FdwFlags.Contains(f_PFD_SUPPORT_GDI)) Temp  << f_PFD_DOUBLEBUFFER;
        else Temp << f_PFD_SUPPORT_GDI;
       }
   FdwFlags=Temp;
   return;
  }
//--------------------------------------------------------------------------

void __fastcall TOpenGLPanel::SetPFDNumColorBits(const BYTE Value)
  {
   FcColorBits=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetPFDNumDepthBits(const BYTE Value)
  {
   FcDepthBits=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetOpenGLPrintScale(const TOpenGLPrintScale Value)
  {
   FOpenGLPrintScale=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetOpenGLPixelsPerInch(const int Value)
  {
   FOpenGLPixelsPerInch=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetFont3DType(TFont* Value)
  {
   Font3DType->Assign(Value);
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetFont3DFormat(const TFont3DFormat Value)
  {
   if (Value==f_WGL_FONT_LINES) SavedFont3DFormat=f_WGL_FONT_LINES;
   else SavedFont3DFormat=WGL_FONT_POLYGONS;
   Font3DFormat=Value;
   return;
  }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SetFont(TFont* Value)
  {
   Font2DType->Assign(Value);
   return;
  }
//---------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Protected Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::WMCreate(TWMCreate &Message)
{
 if (!CreateGLContext())
     {
      ShowMessage("Open GL Context Create Failed");
      return;
     }
 if (FOnInit)
  {
   if (wglMakeCurrent(DisplayDeviceContext,GLRenderingContext)==false)
        ShowMessage("wglMakeCurrent:" + IntToStr((int)GetLastError()));
   if (Font3DEnabled) Create3DFont();
   if (Font2DEnabled) Create2DFont();
   FOnInit(this);
   wglMakeCurrent(NULL, NULL);
  }
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::WMPaletteChanged(TWMPaletteChanged &Message)
{
 TWMQueryNewPalette Dummy;
 if (Message.PalChg==Handle) return;
 WMQueryNewPalette(Dummy);
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::WMQueryNewPalette(TWMQueryNewPalette &Message)
{
  UINT     num;
  HPALETTE OldPalette=SelectPalette(DisplayDeviceContext,Palette,FALSE);
  if ((num=RealizePalette(DisplayDeviceContext))==GDI_ERROR)
        {
         ShowMessage("RealizePalette:"+IntToStr((int)GetLastError()));
         return;
        }
  if (wglMakeCurrent(DisplayDeviceContext,GLRenderingContext)==false)
        {
         Message.Result=0;
         return;
        }
  wglMakeCurrent(NULL, NULL);
  SelectPalette((HDC)GLRenderingContext,OldPalette,TRUE);
  if (num) InvalidateRect(Handle,NULL,TRUE); // Force Repaint
  Message.Result=num;
  return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::WMDestroy(TWMDestroy &Message)
{
  DestroyGLContext();
  return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::WMEraseBkgnd(TWMEraseBkgnd &Message)
{
 if (ComponentState.Contains(csDesigning))
    {
      TRect Rect;
      TColor OldColor=Canvas->Brush->Color;
      Rect =GetClientRect();
      Canvas->Brush->Color=clBtnFace;
      Canvas->FillRect(Rect);
      Canvas->Brush->Color=OldColor;
    }
 Message.Result=TRUE;
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::CreateParams(TCreateParams &Params)
{
 if (FOnPreInit) FOnPreInit(this);
 TCustomPanel::CreateParams(Params);
 Params.Style |= (WS_CLIPCHILDREN | WS_CLIPSIBLINGS);
 Params.ExStyle|= 0;             // Place Holder
 Params.WindowClass.style|=CS_OWNDC;
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Paint(void)
{
  //TCustomPanel::Paint();
  if (FOnPaint)
     {
      HPALETTE OldPalette=NULL;

      if (Palette)
         {
          OldPalette=SelectPalette(DisplayDeviceContext,Palette,FALSE);
          if (RealizePalette(DisplayDeviceContext)==GDI_ERROR)
             {
              ShowMessage("RealizePalette:"+IntToStr((int)GetLastError()));
              return;
             }
        }

      if (wglMakeCurrent(DisplayDeviceContext,GLRenderingContext)==false)
         {
          ShowMessage("wglMakeCurrent:" + IntToStr((int)GetLastError()));
          return;
         }

      FOnPaint(this);

      if ((FSwapBuffers==Auto) && (DoubleBuffer))
            ::SwapBuffers(DisplayDeviceContext);

      if (OldPalette)
           SelectPalette(DisplayDeviceContext,OldPalette,TRUE);

      wglMakeCurrent(NULL, NULL) ;
    }
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Resize(void)
{
  TCustomPanel::Resize();
  if (FOnResize)
     {
      if (wglMakeCurrent(DisplayDeviceContext,GLRenderingContext)==false)
          {
            ShowMessage("wglMakeCurrent:" + IntToStr((int)GetLastError()));
            return;
          }
      FOnResize(this);
      wglMakeCurrent(NULL, NULL) ;
     }
  return;
}
//---------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Public Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::MakeOpenGLPanelCurrent(void)
{
 if (wglMakeCurrent(DisplayDeviceContext,GLRenderingContext)==false)
    {
     ShowMessage("wglMakeCurrent:" + IntToStr((int)GetLastError()));
    }
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::MakeOpenGLPanelNotCurrent(void)
{
  wglMakeCurrent(NULL, NULL);
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SwapBuffers(void)
 {
   if (DoubleBuffer) ::SwapBuffers(DisplayDeviceContext);
 }
//---------------------------------------------------------------------------
bool __fastcall TOpenGLPanel::PanelShareLists(TOpenGLPanel *WithOpenGLPanel)
 {
  return(wglShareLists(this->GLRenderingContext,
                       WithOpenGLPanel->GLRenderingContext));
 }
//---------------------------------------------------------------------------
BMPTexture * __fastcall TOpenGLPanel::LoadBMPTexture(AnsiString FileName,
                                                     Boolean WithAlpha)
{
 Graphics::TBitmap * Bitmap= new Graphics::TBitmap();
 BMPTexture        * BMPTex =new BMPTexture();;
 try
    {
     int  Onedarray;

     Bitmap->LoadFromFile(FileName);
     if (WithAlpha) BMPTex->Components=4;
     else BMPTex->Components=3;
     BMPTex->Width= Bitmap->Width;
     BMPTex->Height=Bitmap->Height;
     BMPTex->Format=WithAlpha?GL_RGBA:GL_RGB;
     BMPTex->Type=GL_UNSIGNED_BYTE;
     BMPTex->Pixels = new GLubyte[BMPTex->Width*BMPTex->Height*BMPTex->Components];
     for (long i=0; i<BMPTex->Height; i++)
       for (long j=0; j<BMPTex->Width; j++)
        {
         Onedarray=i*BMPTex->Width*BMPTex->Components+j*BMPTex->Components;
         BMPTex->Pixels[Onedarray+0] =
                   GetRValue(Bitmap->Canvas->Pixels[j][BMPTex->Height-i-1]);
         BMPTex->Pixels[Onedarray+1] =
                   GetGValue(Bitmap->Canvas->Pixels[j][BMPTex->Height-i-1]);
         BMPTex->Pixels[Onedarray+2] =
                   GetBValue(Bitmap->Canvas->Pixels[j][BMPTex->Height-i-1]);
         if (WithAlpha) BMPTex->Pixels[Onedarray+3] = 255;
        }
     }
 catch (...)
  {
   delete  Bitmap;
   delete  BMPTex;
   return(NULL);
  }
  delete  Bitmap;
  return(BMPTex);

}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::PrintOpenGLPanel(void)
{
 Graphics::TBitmap *BitMap = new Graphics::TBitmap();
 int                XPixelsPerInch,YPixelsPerInch,
                    XSizeInPixels,YSizeInPixels;
 unsigned int       BitMapInfoSize,BitMapImageSize;
 void              *BitMapImage;
 BITMAPINFO        *BitMapInfo;
 RECT               Rect;

 BitMap->Height= Height;
 BitMap->Width = Width;
 BringToFront();
 Paint();
 BitMap->Canvas->CopyRect(ClientRect ,Canvas,ClientRect);
 XPixelsPerInch=GetDeviceCaps(Printer()->Handle,LOGPIXELSX);
 YPixelsPerInch=GetDeviceCaps(Printer()->Handle,LOGPIXELSY);

 Rect.left  =0.18*XPixelsPerInch;
 Rect.top   =0.18*YPixelsPerInch;

 if (FOpenGLPrintScale==pglPrintToFit)
    {
     XSizeInPixels=GetDeviceCaps(Printer()->Handle,HORZRES)-2*Rect.left;
     YSizeInPixels=GetDeviceCaps(Printer()->Handle,VERTRES)-2*Rect.top;
     if (XSizeInPixels > YSizeInPixels)
        {
         Rect.right=BitMap->Width*(YSizeInPixels/BitMap->Height);
         Rect.bottom=YSizeInPixels;
        }
     else
        {
         Rect.right= XSizeInPixels;
         Rect.bottom = BitMap->Height* (XSizeInPixels / BitMap->Width);
        }
    }
 else if (FOpenGLPrintScale==pglProportional)
    {
     if (FOpenGLPixelsPerInch<=0)  FOpenGLPixelsPerInch=96;

     Rect.right =(int)(((float)BitMap->Width/(float)FOpenGLPixelsPerInch)*
                 (float)XPixelsPerInch)+Rect.left;
     Rect.bottom=(int)(((float)BitMap->Height/(float)FOpenGLPixelsPerInch)*
                 (float)YPixelsPerInch)+Rect.top;
    }
  else // pglNone
    {
     Rect.right =BitMap->Width+Rect.left;
     Rect.bottom=BitMap->Height+Rect.right;
    }

 GetDIBSizes(BitMap->Handle,BitMapInfoSize,BitMapImageSize);
 BitMapInfo=(BITMAPINFO *)malloc(BitMapInfoSize);
 BitMapImage=malloc(BitMapImageSize);
 if ((BitMapInfo==NULL) || (BitMapImage==NULL))
  {
   ShowMessage("Print: Malloc Error");
   if (BitMapInfo) free(BitMapInfo);
   if (BitMapImage) free(BitMapImage);
   return;
  }
 Printer()->BeginDoc();
 GetDIB(BitMap->Handle,BitMap->Palette,BitMapInfo,BitMapImage);
 StretchDIBits(Printer()->Canvas->Handle,
              Rect.left,Rect.top,
              Rect.right-Rect.left,
              Rect.bottom-Rect.top,
              0,0,
              BitMap->Width,BitMap->Height,
              BitMapImage,
              BitMapInfo,
              DIB_RGB_COLORS,
              SRCCOPY);
 free(BitMapInfo);
 free(BitMapImage);
 Printer()->EndDoc();
 delete BitMap;
 return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::SaveOpenGLPanelToFile(const String &FileName)
{
 Graphics::TBitmap *BitMap = new Graphics::TBitmap();
 BitMap->Height= Height;
 BitMap->Width = Width;
 BringToFront();
 Paint();
 BitMap->Canvas->CopyRect(ClientRect ,Canvas,ClientRect);
 BitMap->SaveToFile(FileName);
 delete BitMap;
 return;
}
//---------------------------------------------------------------------------
OpenGLFont3D * __fastcall TOpenGLPanel::Create3DFont(TFont *Font,
                                                     int FirstGylph,
                                                     int NumGylph,
                                                     float Deviation,
                                                     float Extrusion,
                                                     int Format)
{
 HFONT         HFont;
 LOGFONT       LogFont;
 GLuint        Base;

 if (Font==NULL) return(NULL);
 if ((Base=glGenLists(NumGylph))==0) return(NULL);
 OpenGLFont3D *Font3D=new OpenGLFont3D();
 Font3D->FirstGylph=FirstGylph;
 Font3D->NumGylph=NumGylph;
 Font3D->ListBase=Base;
 Font3D->Extrusion=Extrusion;
 Font3D->GlyphMetrics= new GLYPHMETRICSFLOAT[NumGylph];
 GetObject(Font->Handle,sizeof(LOGFONT),&LogFont);
  //  FYI To go from LogFont -> Handle use:
  //  Font->Handle =CreateFontIndirect(LogFont);
 HFont = CreateFontIndirect(&LogFont);
 SelectObject (DisplayDeviceContext, HFont);
 wglUseFontOutlines(DisplayDeviceContext, FirstGylph, NumGylph, Base,
                      Deviation, Extrusion, Format, Font3D->GlyphMetrics);
 DeleteObject(HFont);
 return(Font3D);
 }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Get3DTextSize(AnsiString Text,GLdouble *XYZ)
{
  Get3DTextSize(Font3DDefault,Text,XYZ);
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Get3DTextSize(OpenGLFont3D *Font,
                                            AnsiString Text,GLdouble *XYZ)
{
 unsigned char *c=(unsigned char *)Text.c_str();
 int          Index,Length=strlen((char *)c);

 if ((Font==NULL)||(XYZ==NULL)||(Length==0)) return;
 XYZ[0]=0.0;
 XYZ[1]=0.0;
 XYZ[2]=Font->Extrusion;
 for (int i=0;i<Length;i++)
  {
   Index=c[i]- Font->FirstGylph;
   if ((Index<0)||(Index>Font->NumGylph-1))
   {
    XYZ[0]=XYZ[1]=XYZ[2]=0.0;
    return;
   }
   XYZ[0]+=Font->GlyphMetrics[Index].gmfCellIncX;
 //  XYZ[1]+=Font->GlyphMetrics[Index].gmfCellIncY;

  }
  Index=c[Length-1]- Font->FirstGylph;
  if ((Index<0)||(Index>Font->NumGylph-1))
   {
    XYZ[0]=XYZ[1]=XYZ[2]=0.0;
    return;
   }
 // XYZ[0]+=Font->GlyphMetrics[Index].gmfBlackBoxX;
  XYZ[1]+=Font->GlyphMetrics[Index].gmfBlackBoxY;
  return;
}
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Draw3DText(AnsiString Text)
 {
  Draw3DText(Font3DDefault,Text);
 }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Draw3DText(OpenGLFont3D *Font,AnsiString Text)
{
  GLint  FaceMode;
  unsigned char *c=(unsigned char *)Text.c_str();
  int            Index,Length=strlen((char *)c);

  if ((Font==NULL)||(Length==0)) return;
  for (int i=0;i<Length;i++)
  {
   Index=c[i]- Font->FirstGylph;
   if ((Index<0)||(Index>Font->NumGylph-1)) return;
  }
  glPushAttrib(GL_LIST_BIT);
  glGetIntegerv(GL_FRONT_FACE,&FaceMode);
  glListBase(Font->ListBase-Font->FirstGylph);
  glCallLists (Length, GL_UNSIGNED_BYTE,c);
  glFrontFace(FaceMode);
  glPopAttrib();
  return;
}
//---------------------------------------------------------------------------
OpenGLFont2D * __fastcall TOpenGLPanel::Create2DFont(TFont *Font,
                                                     int FirstGylph,
                                                     int NumGylph)
{
 HFONT         HFont;
 LOGFONT       LogFont;
 GLuint        Base;

 if (Font==NULL) return(NULL);
 if ((Base=glGenLists(NumGylph))==0) return(NULL);
 OpenGLFont2D *Font2D=new OpenGLFont2D();
 Font2D->FirstGylph=FirstGylph;
 Font2D->NumGylph=NumGylph;
 Font2D->ListBase=Base;
 GetObject(Font->Handle,sizeof(LOGFONT),&LogFont);
  //  FYI To go from LogFont -> Handle use:
  //  Font->Handle =CreateFontIndirect(LogFont);
 HFont = CreateFontIndirect(&LogFont);
 SelectObject (DisplayDeviceContext, HFont);
 wglUseFontBitmaps(DisplayDeviceContext, FirstGylph, NumGylph, Base);
 DeleteObject(HFont);
 return(Font2D);
 }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Draw2DText(AnsiString Text)
 {
  Draw2DText(Font2DDefault,Text);
 }
//---------------------------------------------------------------------------
void __fastcall TOpenGLPanel::Draw2DText(OpenGLFont2D *Font,AnsiString Text)
{
  unsigned char *c=(unsigned char *)Text.c_str();
  int            Index,Length=strlen((char *)c);

  if ((Font==NULL)||(Length==0)) return;
  for (int i=0;i<Length;i++)
  {
   Index=c[i]- Font->FirstGylph;
   if ((Index<0)||(Index>Font->NumGylph-1)) return;
  }
  glPushAttrib(GL_LIST_BIT);
  glListBase(Font->ListBase-Font->FirstGylph);
  glCallLists (Length, GL_UNSIGNED_BYTE,c);
  glPopAttrib();
  return;
}
//---------------------------------------------------------------------------
bool   __fastcall TOpenGLPanel::QueryOpenGLExtention(AnsiString ExtentionName)
{
 char *GLExtentions,*LastChar,*ExtQName=ExtentionName.c_str();
 int   ExtLen,ExtQNameLen= strlen(ExtQName);
 if ((GLExtentions =(char *)glGetString(GL_EXTENSIONS))==NULL) return(false);

 LastChar = GLExtentions + strlen(GLExtentions);

 while (GLExtentions < LastChar)
  {
   ExtLen= strcspn(GLExtentions, " ");
   if ((ExtQNameLen == ExtLen) && (strncmp(ExtQName,GLExtentions,ExtLen) == 0))
      return true;
   GLExtentions += (ExtLen + 1);
  }
  return false;
}
//---------------------------------------------------------------------------
int   __fastcall TOpenGLPanel::GetDisplayBPP(void)
{
 int BPP;
 HDC DeviceContext=GetDC(NULL);
 BPP=GetDeviceCaps(DeviceContext,BITSPIXEL);
 ReleaseDC(NULL,DeviceContext);
 return(BPP);
}
//---------------------------------------------------------------------------
bool   __fastcall TOpenGLPanel::SetDisplayBPP(int BPP)
{
  // Warning:: this code does not work correctly
  // It was intended to be executed in the OnPreInit Event
  // The display mode changes but then GL Initializaion Fails??
  DEVMODE LPDevMode;
  ZeroMemory(&LPDevMode,sizeof(LPDevMode));
  LPDevMode.dmSize=sizeof(LPDevMode);
  LPDevMode.dmBitsPerPel=BPP;
  LPDevMode.dmFields=DM_BITSPERPEL;
  return(ChangeDisplaySettings(&LPDevMode,0)==DISP_CHANGE_SUCCESSFUL?true:false);
}
//---------------------------------------------------------------------------
