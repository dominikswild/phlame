"""Defines classes that describe the geometry of the stack.

MIT License

Copyright (c) 2019 Dominik S. Wild

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings


class Stack():
    """Defines stack in terms of layers and patters.

    Attributes:
        lattice_constant: Lattice constant of the periodic structure.
        material_dict: A dictionary containing user defined materials.
        pattern_dict: A dictionary containing user defined patterns.
        top_layer: The top layer of the stack.
    """

    def __init__(self, lattice_constant=None):
        """Initializes Stack instance. The lattice constant can be set (or
        changed) at a later point with set_lattice_constant."""
        self.lattice_constant = lattice_constant
        self.material_dict = {}
        self.pattern_dict = {}
        self.top_layer = None


    def __validate_pattern(self, pattern_name):
        """Checks if 'pattern_name' exists in 'pattern_dict'. If it doesn't but
        'material_dict' has a corresponding entry, a new homogenous pattern
        made of material 'pattern_name' with 'pattern_name' is created.

        Raises:
            ValueError: 'pattern_name' does not refer to an existing pattern or
                material.
        """
        if pattern_name not in self.pattern_dict:
            if pattern_name not in self.material_dict:
                raise ValueError(f"Pattern or material name '{pattern_name}' "
                                 "not found.")
            self.define_pattern(pattern_name, pattern_name)


    def set_lattice_constant(self, lattice_constant):
        """Sets lattice constant and clears cache of stack.

        Args:
            lattice_constant: Lattice constant.
        """
        self.lattice_constant = lattice_constant
        self.clear_cache()


    def define_material(self, material_name, permittivity,
                        two_dimensional=None):
        """Adds new material to dictionary. If the material already exists, its
        properties are updated. Note that existing two-dimensional materials
        cannot be converted into three-dimensional ones or vice versa.

        Args:
            material_name: String specifying material name.
            permittivity: The permittivity defined as a number or a 3-element
                list/tuple/array of numbers. If it is a number, the material is
                taken to be isotropic. If it is a list, then the three elements
                correspond to the diagonal components of the permittivity
                tensor. Materials whose major axes are not aligned with the
                coordinate axes are currently not supported.

        Raises:
            RuntimeError: User attempted to convert 2D material in to 3D
                material or vice versa.
        """
        if material_name in self.material_dict:
            material = self.material_dict[material_name]
            if two_dimensional is not None:
                if two_dimensional != material.two_dimensional:
                    raise RuntimeError("It is not possible to convert a 2D "
                                       "material into a 3D one or vice versa.")
            material.set_permittivity(permittivity)
            self.clear_cache(material)
        else:
            self.material_dict[material_name] = Material(permittivity,
                                                         two_dimensional)


    def define_pattern(self, pattern_name, material_name_list, width_list=None):
        """Adds new pattern to dictionary. If pattern already exists, its
        properties are updated and its cache is cleared.

        Args:
            pattern_name: String specifying pattern name.
            material_name_list: A list or tuple of material names. If
                width_list is None, material_name_list can be a string instead
                of a list of strings.
            width_list: A list or tuple of positive numbers corresponding to
                width of each material in material_name_list. The sum of the
                widths is irrelevant as it is normalized to 1, making each
                pattern independent of the lattice constant. If width_list is
                omitted, a homogeneous pattern made of the first material in
                material_name_list is created.
        """
        if width_list is None:
            if isinstance(material_name_list, str):
                material_list = [self.material_dict[material_name_list]]
            else:
                material_list = [self.material_dict[material_name_list[0]]]
            width_list = [1]
        else:
            material_list = [self.material_dict[material_name]
                             for material_name in material_name_list]

        if pattern_name in self.pattern_dict:
            pattern = self.pattern_dict[pattern_name]
            pattern.material_list = material_list
            pattern.width_list = width_list
            pattern.clear_cache()
        else:
            self.pattern_dict[pattern_name] = Pattern(material_list, width_list)


    def add_layer(self, pattern_name, thickness):
        """Prepends a single layer to the stack. It first looks for
        'pattern_name' in 'pattern_dict'. If no pattern with this name exists
        but the name matches a material in 'material_dict', the function
        creates a new homogeneous pattern for the material and uses it for the
        layer.

        Args:
            pattern_name: String with the name of the pattern or material to be
                added.
            thickness: Number specifying the thickness of the layer.
        """
        self.__validate_pattern(pattern_name)
        pattern = self.pattern_dict[pattern_name]
        layer = Layer(pattern, thickness)
        layer.next = self.top_layer
        self.top_layer = layer


    def add_layers(self, pattern_list, thickness_list):
        """Prepends multiple layers to the stack. The patterns and thicknesses
        are provided as lists or tuples, where the first entry corresponds to
        the resulting top layer.

        Args:
            pattern_list: List or tuple of strings containing the names of the
                patterns to be added. A single string can be used to add a
                single layer, though add_layer is preferred in this case.
            thickness_list: List or tuple of numbers specifying the thickness
                of each layer. A single number can be used to add a single
                layer, though add_layer is preferred in this case.

        Raises:
            ValueError: 'pattern_list' and 'thickness_list' do not have the
                same length.
        """
        if isinstance(pattern_list, str):
            self.add_layer(pattern_list, thickness_list)
            return

        if len(pattern_list) != len(thickness_list):
            raise ValueError("'pattern_list' must have the same length as "
                             "'thickness_list'")

        for pattern_name, thickness in zip(reversed(pattern_list),
                                           reversed(thickness_list)):
            self.add_layer(pattern_name, thickness)


    def set_layer_thickness(self, index, thickness):
        """Sets the thickness of the layer specified by index.

        Args:
            index: Integer specifying the layer. The layers are numbered from
                top to bottom, with the top layer having index 0.
            thickness: New thickness of the layer.

        Raises:
            ValueError: The index is outside the valid range.
            Warning: The thickness of the bottom or a two-dimensional layer was
                changed.
        """
        layer = self.top_layer
        cur_index = 0
        while layer.next and index > cur_index:
            layer = layer.next
            cur_index += 1

        if index > cur_index:
            raise ValueError(f"(index = {index}) exceeds "
                             f"(number of layers - 1 = {cur_index})")
        if not layer.next:
            warnings.warn(
                "Changing the thickness of the bottom layer has no effect."
            )
        if layer.pattern.two_dimensional:
            warnings.warn(
                "Changing the thickness of a two-dimensional layer has no "
                "effect."
            )

        layer.thickness = thickness


    def set_layer_pattern(self, index, pattern_name):
        """Sets the pattern of the layer specified by index.

        Args:
            index: Integer specifying the layer. The layers are numbered from
                top to bottom, with the top layer having index 0.
            pattern: New pattern of the layer.

        Raises:
            ValueError: The index is outside the valid range.
        """
        layer = self.top_layer
        cur_index = 0
        while layer.next and index > cur_index:
            layer = layer.next
            cur_index += 1

        if index > cur_index:
            raise ValueError(f"(index = {index}) exceeds "
                             f"(number of layers - 1 = {cur_index})")

        self.__validate_pattern(pattern_name)
        layer.pattern = self.pattern_dict[pattern_name]


    def clear_cache(self, *args):
        """Clears the cache of patterns.

        Args:
            *args: If the function is called without an argument, the cache of
            all patterns in the stack is cleared. The function can be called
            with a material as an argument, in which case the cache of all
            patterns containing the material is cleared.
        """
        if not args:
            for _key, pattern in self.pattern_dict.items():
                pattern.clear_cache()
        else:
            for _key, pattern in self.pattern_dict.items():
                if args[0] in pattern.material_list:
                    pattern.clear_cache()


    def print(self):
        """Prints information about all layers in the stack."""
        layer = self.top_layer
        layer_index = 0
        while layer:
            print(f"Layer {layer_index}:")

            print(f"\tTickness: {layer.thickness}", end="")
            if (layer_index == 0 or
                    layer.next is None or layer.pattern.two_dimensional):
                print(" (unused)", end="")
            print()

            lattice_constant = self.lattice_constant
            if lattice_constant is None:
                lattice_constant = 1
            width_string = "\tWidths: ["
            for i, width in enumerate(layer.pattern.width_list):
                if width is None:
                    width_string += "None"
                else:
                    width_string += f"{lattice_constant*width:.3g}"
                if i < len(layer.pattern.width_list) - 1:
                    width_string += ", "
            width_string += "]"
            print(width_string)

            # permittivity string is more complicated to handle complex numbers
            permittivity_list = [material.permittivity
                                 for material in layer.pattern.material_list]
            permittivity_string = "\tPermittivities: ["
            for i, permittivity in enumerate(permittivity_list):
                permittivity_string += "("
                for j, permittivity_i in enumerate(permittivity):
                    permittivity_string += f"{permittivity_i:.3g}"
                    if j < len(permittivity)-1:
                        permittivity_string += ", "
                permittivity_string += ")"
                if i < len(permittivity_list)-1:
                    permittivity_string += ", "
            permittivity_string += "]"
            print(permittivity_string)

            print()
            layer = layer.next
            layer_index += 1



class Pattern():  # pylint: disable=too-few-public-methods
    """Defines an in-plane periodic pattern. The class also enables caching for
    propagation through the pattern.

    Attributes:
        material_list: A list of Material instances.
        width_list: A list of widths corresponding to each material. The widths
            are normalized such that they sum to 1.
        two_dimensional: True if the pattern is made up of two-dimensional
            materials, False otherwise.
        homogeneous: Flag that signals whether the pattern is homogeneous.
        cache: Dictionary used by core functions to cache computation results.
    """

    def __init__(self, material_list, width_list):
        """
        Raises:
            RuntimeError: User attempted to create a pattern from a combination
                of two-dimensional and three-dimensional materials.
        """
        for material in material_list:
            if material.two_dimensional != material_list[0].two_dimensional:
                raise RuntimeError("It is not possible to combine "
                                   "two-dimensional and three-dimensional "
                                   "materials in a pattern.")
        self.two_dimensional = material_list[0].two_dimensional

        self.material_list = list(material_list)

        width_total = sum(width_list)
        self.width_list = [width/width_total for width in width_list]

        if len(self.material_list) == 1:
            self.homogeneous = True
        else:
            self.homogeneous = False

        self.cache = {}


    def clear_cache(self):
        """Clear the cache of the pattern."""
        self.cache = {}



class Layer():  # pylint: disable=too-few-public-methods
    """Defines layer in terms of a pattern and a thickness. It is implemented
    as a class to allow for a linked list structure, where each layer links to
    the layer directly below.

    Attributes:
        pattern: Instance of Pattern, specifying the in-plane pattern of the
            layer.
        thickness: Number specifying the thickness of the layer.
        next: Instance of Layer, pointing to the layer directly below the
            current one.
    """

    def __init__(self, pattern, thickness):
        self.pattern = pattern
        self.thickness = thickness
        self.next = None



class Material():   # pylint: disable=too-few-public-methods
    """Instances of this class are used to store information about each
    material. It is implemented as a class to allow for more complicated
    features in the future such as frequency dependence.

    Attributes:
        permittivity: 3-element list of permittivities. We assume that the
            major axes are aligned with the coordinate axes such that the three
            components correspond to the diagonal components of the
            permittivity tensor.
        two_dimensional: A flag which True if the material is infinitesimally
            thick. In this case, permittivity is the 2D permittivity, which has
            units of length.
        isotropic: True if the material is isotropic within the plane, False
            otherwise. More accurately, if this flag is True, the material is
            uniaxial with the symmetry axis aligned with the normal to the
            layers.
    """

    def __init__(self, permittivity, two_dimensional):
        self.set_permittivity(permittivity)
        self.two_dimensional = two_dimensional


    def set_permittivity(self, permittivity):
        """Distinguishes between scalar and list input and sets the
        permittivity accordingly.

        Args:
            permittivity: A number of a 3-element list/tuple/array of numbers.

        Raises:
            ValueError: Invalid format.
        """
        if isinstance(permittivity, (int, float, complex)):
            self.permittivity = [permittivity for i in range(0, 3)]
            self.isotropic = True
        else:
            if len(permittivity) != 3:
                raise ValueError("'permittivity' must be a scalar or a "
                                 "three-element list/tuple/array.")

            self.permittivity = list(permittivity)

            if permittivity[0] == permittivity[1]:
                self.isotropic = True
            else:
                self.isotropic = False
