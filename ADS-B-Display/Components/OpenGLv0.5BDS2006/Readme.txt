OpenGLPanel.cpp Version 0.5BDS2006 ALPHA


This OpenGL Component is for C++ Builder Version 5.0.

Uninstall Procedure
===================
1. Under the file menu select "close all files".
2. Under the "Component" menu Item, select "Install Packages".
3. In the "Install Packages" dialog select the
   "OpenGL Panel Component" in the "Design Packages" section.
4. Uncheck the "OpenGL Panel Component" and ensure that it is
   still selected.
5. Click on the "Remove" button.
6. Click on "Yes" in the Confirm Dialog.
7. Click on "OK" in the "Install Packages" dialog. 
8. Exit C++ Builder.



Component Installation
=======================
To install the OpenGL Panel component, follow the steps outlined below. 


 1. Unzip the distribution file in to a directory.

 2. Go to the component directory of the package.

 3. Double click on OpenGLPanel_DP.bpk  (this will launch C++ Builder 5.0)

 4. Under the projects menu select Build OpenGLPanel_DP.

 5. In the package window, click the install button.

 6. Note that the component has been installed and press ok.   

 7. Under the file menu select "close all files".
 
 8. Answer Yes to the question "Save changes to OpenGLPanel_DP".
 
 9. The OpenGL Panel Component is now installed.

10. Set your include path in BCB5 to include the path to the component source files.
          
       include example:
       $(BCB)\include;$(BCB)\include\vcl;c:\dplakosh\openglv0.5bcb5\component.

11. Set your lib path in BCB5 so to include the installed location of the OpenGL lib and bpl files.
    These files are usually installed at location $(BCB)\Projects\Lib
       lib example:
        $(BCB)\lib\obj;$(BCB)\lib;$(BCB)\Projects\Lib
      
 

 The demo programs serve as use examples.

Contact Daniel Plakosh at <dplakosh@cobweb.net> or
<dplakosh@sei.example.org> to request assistance or to report a problem with
this software. Criticism, suggestions and comments are also greatly
appreciated. Remember to always include the OpenGLPanel version and system
information in your messages.
