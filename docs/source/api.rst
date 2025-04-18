API Documentation
=================

This section provides detailed API documentation for the individual modules within the `stop-utils` package.

Command-Line Interface (`cli`)
------------------------------

.. automodule:: stop_utils.cli
   :members:
   :undoc-members:
   :show-inheritance:

WFE Analysis (`wfe_analysis`)
-----------------------------

.. automodule:: stop_utils.wfe_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Visualization (`visualization`)
-------------------------------

.. automodule:: stop_utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:

File Format Converters (`converters`)
-------------------------------------

.. automodule:: stop_utils.converters
   :members:
   :undoc-members:
   :show-inheritance:

Types (`types`)
---------------

.. automodule:: stop_utils.types
   :members:
   :undoc-members:
   :show-inheritance:

Zemax Integration (`zemax`)
---------------------------

.. note::
   The Zemax integration modules require a Windows system with Zemax OpticStudio installed.
   These modules use the Zemax ZOS-API which is only available on Windows platforms.

Prerequisites:
    * Windows operating system
    * Zemax OpticStudio (Premium, Professional, or Standard Edition)
    * Valid Zemax license for API use
    * Python.NET (pythonnet) package installed

The Zemax submodule provides integration with Zemax OpticStudio through its ZOS-API.

Core ZOS-API Interface (`zemax.zemax_wfe`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: stop_utils.zemax.zmx_boilerplate
  :members:
  :undoc-members:
  :show-inheritance:

Wavefront Extractor (`zemax.wavefront_extractor`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: stop_utils.zemax.wavefront_extractor
  :members:
  :undoc-members:
  :show-inheritance:

Batch Processor (`zemax.zmx_batch_processor`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: stop_utils.zemax.zmx_batch_processor
  :members:
  :undoc-members:
  :show-inheritance:

.. warning::
  Attempting to use these modules on non-Windows platforms will result in
  import errors for Windows-specific dependencies like 'winreg' and the Zemax API components.