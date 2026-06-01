:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ fullname | replace("surface_sim.", "surface_sim::") }}

{# In the fullname, the module name is ambiguous. Using a `::` separator
specifies `surface_sim` as the module name. #}
