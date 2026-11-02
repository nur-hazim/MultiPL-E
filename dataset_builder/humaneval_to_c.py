"""HumanEval C11 signatures, literals and equality tests.

Lists/dictionaries are pointer+length structs; tuples are product structs.
Optional/Union/Any use a tagged value. Integers are signed 64 bit. Returned
storage must remain valid after return. Python examples remain in comments.
"""
import ast
from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class Expr:
    kind: str
    value: object

    def __str__(self):
        return repr(self)


def annotation(node):
    if isinstance(node, ast.Name) and node.id in ('int', 'float', 'bool', 'str'):
        return (node.id,)
    if isinstance(node, ast.Name) and node.id == 'Any':
        return ('value',)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        name = node.value.id
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name in ('Optional', 'Union'):
            return ('value',)
        if name == 'Tuple' and isinstance(args[-1], ast.Constant) and args[-1].value is Ellipsis:
            return ('list', annotation(args[0]))
        if name in ('List', 'Tuple', 'Dict'):
            return (name.lower(), *(annotation(x) for x in args))
    raise ValueError(f'Unsupported C annotation: {ast.unparse(node)}')


VALUE = '''
/* he_value tags: 0=None, 1=bool, 2=int, 3=float, 4=UTF-8 string,
   5=list, 6=tuple, 7=dict (parallel keys/items). */
typedef struct he_value he_value;
struct he_value {
    int tag; int64_t integer; double real; char *string;
    size_t length; he_value *items; he_value *keys;
};
'''


class Translator:
    # No textual stop token: a closing brace may belong to a nested block or
    # helper function. Generation is bounded by max_tokens instead.
    stop = []

    def __init__(self):
        self.types = {}
        self.args = []
        self.ret = ('int',)
        self.entry = ''

    def file_ext(self):
        return 'c'

    def ctype(self, spec):
        primitive = {'int': 'int64_t', 'float': 'double', 'bool': 'bool',
                     'str': 'char *', 'value': 'he_value'}
        if spec[0] in primitive:
            return primitive[spec[0]]
        if spec not in self.types:
            children = [self.ctype(s) for s in spec[1:]]
            name = 'he_' + spec[0] + '_' + hashlib.sha256(repr(spec).encode()).hexdigest()[:10]
            if spec[0] == 'list':
                fields = f'{children[0]} *data; size_t length;'
            elif spec[0] == 'tuple':
                fields = ' '.join(f'{c} item{i};' for i, c in enumerate(children))
            elif spec[0] == 'dict':
                fields = f'{children[0]} *keys; {children[1]} *values; size_t length;'
            else:
                raise ValueError(spec)
            self.types[spec] = (name, f'typedef struct {{ {fields} }} {name};')
        return self.types[spec][0]

    def header(self):
        return ('#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n'
                '#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n'
                '#include <assert.h>\n' + VALUE + '\n'
                + '\n'.join(decl for _, decl in self.types.values()) + '\n')

    def translate_prompt(self, name, args, returns, description):
        self.types = {}
        self.entry = name
        self.args = [annotation(a.annotation) for a in args]
        # HumanEval 92's float annotations are misleading: the task explicitly
        # distinguishes Python int(3) from float(3.0). Preserve the runtime tag.
        if name == 'any_int':
            self.args = [('value',)] * len(args)
        self.ret = annotation(returns)
        # Prefix function/argument names to avoid C keywords and libc collisions.
        params = ', '.join(f'{self.ctype(s)} he_arg_{i}' for i, s in enumerate(self.args)) or 'void'
        ret = self.ctype(self.ret)
        comments = '\n'.join('// ' + line for line in description.splitlines())
        bindings = '\n'.join(f'// he_arg_{i}: Python parameter {arg.arg} ({ast.unparse(arg.annotation)})'
                             for i, arg in enumerate(args))
        return (self.header() + '\n// C11 interface: int64_t integers; UTF-8 NUL-terminated strings.\n'
                '// Containers carry lengths. Input storage is borrowed; do not free it.\n'
                '// Return storage must survive the call (heap/static); never return stack pointers.\n'
                '// Do not provide main() or tests. Implement the function and any helpers in C.\n'
                + comments + '\n' + bindings + f'\n{ret} he_{name}({params}) {{\n')

    def gen_literal(self, value):
        return Expr('literal', value)

    def gen_var(self, name):
        return Expr('var', name)

    def gen_list(self, values):
        return Expr('literal', [self.pyvalue(v) for v in values])

    def gen_tuple(self, values):
        return Expr('literal', tuple(self.pyvalue(v) for v in values))

    def gen_dict(self, keys, values):
        return Expr('literal', dict(zip(map(self.pyvalue, keys), map(self.pyvalue, values))))

    def gen_set(self, values):
        raise ValueError('C set translation is not implemented')

    def pyvalue(self, expr):
        if expr.kind != 'literal':
            raise ValueError(f'Expected a constant expression: {expr}')
        return expr.value

    def gen_call(self, func, args):
        if func.kind == 'var' and func.value in ('candidate', self.entry):
            if len(args) != len(self.args):
                raise ValueError('Argument count mismatch')
            return Expr('call', tuple(args))
        if func.kind == 'var' and func.value in ('str', 'len'):
            if len(args) != 1:
                raise ValueError('Builtin argument count mismatch')
            return Expr('literal', {'str': str, 'len': len}[func.value](self.pyvalue(args[0])))
        raise ValueError(f'Unsupported call: {func}')

    def literal(self, value, spec):
        kind = spec[0]
        ty = self.ctype(spec)
        if kind == 'int':
            # The Python reference for tri returns integral floats despite its
            # List[int] annotation; Python equality accepts those as integers.
            if isinstance(value, float) and math.isfinite(value) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or not -(2**63) <= value < 2**63:
                raise ValueError(f'Not an int64_t: {value!r}')
            if value == -(2**63):
                return '(-INT64_C(9223372036854775807)-1)'
            return f'INT64_C({value})' if value >= 0 else f'(-INT64_C({-value}))'
        if kind == 'float':
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Not a finite double: {value!r}')
            return repr(float(value))
        if kind == 'bool':
            if not isinstance(value, bool):
                raise ValueError(f'Not a bool: {value!r}')
            return 'true' if value else 'false'
        if kind == 'str':
            if not isinstance(value, str) or '\0' in value:
                raise ValueError('Expected a string without embedded NUL')
            # Octal escapes preserve UTF-8 bytes without hex-escape ambiguity.
            return '"' + ''.join(f'\\{b:03o}' for b in value.encode('utf-8')) + '"'
        if kind == 'value':
            if value is None:
                fields = '.tag=0'
            elif isinstance(value, bool):
                fields = f'.tag=1,.integer={int(value)}'
            elif isinstance(value, int):
                fields = f'.tag=2,.integer={self.literal(value, ("int",))}'
            elif isinstance(value, float):
                fields = f'.tag=3,.real={self.literal(value, ("float",))}'
            elif isinstance(value, str):
                fields = f'.tag=4,.string={self.literal(value, ("str",))}'
            elif isinstance(value, (list, tuple, dict)):
                vals = list(value.values()) if isinstance(value, dict) else value
                tag = 7 if isinstance(value, dict) else 6 if isinstance(value, tuple) else 5
                fields = f'.tag={tag},.length={len(vals)},.items={self.array(vals, ("value",))}'
                if isinstance(value, dict):
                    fields += f',.keys={self.array(list(value), ("value",))}'
            else:
                raise ValueError(value)
        elif kind == 'list':
            if not isinstance(value, (list, tuple)):
                raise ValueError(value)
            fields = f'.data={self.array(value, spec[1])},.length={len(value)}'
        elif kind == 'tuple':
            if not isinstance(value, tuple) or len(value) != len(spec) - 1:
                raise ValueError('Tuple arity/type mismatch')
            fields = ','.join(f'.item{i}={self.literal(v, s)}' for i, (v, s) in enumerate(zip(value, spec[1:])))
        elif kind == 'dict':
            if not isinstance(value, dict):
                raise ValueError(value)
            fields = (f'.keys={self.array(list(value), spec[1])},'
                      f'.values={self.array(list(value.values()), spec[2])},.length={len(value)}')
        else:
            raise ValueError(spec)
        return f'(({ty}){{{fields}}})'

    def array(self, values, spec):
        if not values:
            return 'NULL'
        return '(' + self.ctype(spec) + '[]){' + ','.join(self.literal(v, spec) for v in values) + '}'

    def render(self, expr):
        if expr.kind == 'call':
            return f'he_{self.entry}(' + ','.join(self.literal(self.pyvalue(a), s) for a, s in zip(expr.value, self.args)) + ')'
        return self.literal(self.pyvalue(expr), self.ret)

    def finalize(self, expr, side):
        return self.render(expr)

    def eqname(self, spec):
        return 'he_eq_' + hashlib.sha256(repr(spec).encode()).hexdigest()[:10]

    def equality(self, spec):
        kind = spec[0]
        if kind in ('int', 'float', 'bool'):
            body = 'return a == b;'
        elif kind == 'str':
            body = 'return a && b && strcmp(a,b)==0;'
        elif kind == 'value':
            body = '''
if (a.tag != b.tag) {
    if (a.tag>=1 && a.tag<=3 && b.tag>=1 && b.tag<=3)
        return (a.tag==3?a.real:a.integer)==(b.tag==3?b.real:b.integer);
    return false;
}
switch(a.tag) {
case 0: return true;
case 1: case 2: return a.integer==b.integer;
case 3: return a.real==b.real;
case 4: return a.string && b.string && strcmp(a.string,b.string)==0;
case 5: case 6: case 7:
    if(a.length!=b.length || a.length>1000000) return false;
    for(size_t i=0;i<a.length;i++) {
        if(a.tag==7) {
            bool found=false;
            for(size_t j=0;j<b.length;j++)
                if(HE_EQ(a.keys[i],b.keys[j]) && HE_EQ(a.items[i],b.items[j])) {found=true;break;}
            if(!found) return false;
        } else if(!HE_EQ(a.items[i],b.items[i])) return false;
    }
    return true;
default: return false;
}'''.replace('HE_EQ', self.eqname(spec))
        elif kind == 'tuple':
            body = 'return ' + ' && '.join(f'{self.eqname(s)}(a.item{i},b.item{i})' for i, s in enumerate(spec[1:])) + ';'
        else:
            body = 'if(a.length!=b.length || a.length>1000000) return false;\n'
            if kind == 'list':
                body += f'for(size_t i=0;i<a.length;i++) if(!{self.eqname(spec[1])}(a.data[i],b.data[i])) return false; return true;'
            else:
                body += (f'for(size_t i=0;i<a.length;i++) {{ bool found=false; for(size_t j=0;j<b.length;j++) '
                         f'if({self.eqname(spec[1])}(a.keys[i],b.keys[j]) && '
                         f'{self.eqname(spec[2])}(a.values[i],b.values[j])) {{found=true;break;}} '
                         'if(!found) return false;} return true;')
        ty = self.ctype(spec)
        return f'static bool {self.eqname(spec)}({ty} a, {ty} b) {{ {body} }}'

    def test_suite_prefix_lines(self, entry_point):
        specs = []
        def collect(s):
            for child in s[1:]:
                collect(child)
            if s not in specs:
                specs.append(s)
        for spec in [*self.args, self.ret]:
            collect(spec)
        return ['#undef NDEBUG\n#include <assert.h>', *(self.equality(s) for s in specs), 'int main(void) {']

    def deep_equality(self, left, right):
        return f'assert({self.eqname(self.ret)}({left}, {right}));'

    def test_suite_suffix_lines(self):
        return ['return 0;', '}']

    def no_completion_prompt_stub(self):
        return f'return ({self.ctype(self.ret)}){{0}};\n}}\n'
