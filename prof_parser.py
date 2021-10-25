import sqlite3
import pandas as pd
import os


class NvvpParser:
    def __init__(self, prof_file, with_nvtx=True, demangle=True,
                 set_time_offset=True, nvprof=True, micro_sec=True):
        self.path, file_name = os.path.split(os.path.abspath(prof_file))
        self.name = os.path.splitext(file_name)[0]
        self.with_nvtx = with_nvtx
        self.demangle = demangle
        self.micro_sec = micro_sec
        self.set_time_offset = set_time_offset

        self.conn = sqlite3.connect(prof_file)
        self.conn.row_factory = sqlite3.Row
        self.time_offset = 0
        self.pd_ops = None
        self.pd_launch_and_execute = None
        self.pd_kernel_level = None
        self.pd_op_level = None
        self.pd_op_avg = None

        if nvprof:
            self.process()

    def get_time_offset(self):
        # Find time zero point of the file from *_DRIVER
        for row in self.conn.execute('SELECT start FROM CUPTI_ACTIVITY_KIND_DRIVER WHERE cbid == 332'):
            self.time_offset = row['start']
            break
        else:
            print('Failed to get time offset.')

    def demangle_kernel_names(self):
        self.pd_launch_and_execute['kernel_name'] = self.pd_launch_and_execute['kernel_name'].map(demangle)

    def tag_steps(self):
        if self.pd_ops is not None and 'step' not in self.pd_ops:
            ops = self.pd_ops['operator'].to_numpy().tolist()
            indices = [i for i, x in enumerate(ops) if x == ops[0]] + [len(ops)]
            step = []
            for i in range(len(indices) - 1):
                step += [i] * (indices[i + 1] - indices[i])
            assert len(step) == len(ops)
            self.pd_ops.insert(0, 'step', step)

    def extract_ort_operators(self):
        '''
        Extract ORT operator marker information.

        Columns in operators (pd_ops):
        - step: step in training loop
        - operator ('Batch- Forward' and 'Batch- Backward' excluded)
        - op_start
        - op_end
        '''
        self.conn.execute('CREATE TEMP TABLE markers AS '
          'SELECT strings.value AS operator, marker_start.timestamp AS op_start, marker_end.timestamp AS op_end '
          'FROM (SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name != 0) AS marker_start '
          'LEFT JOIN (SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name = 0) AS marker_end ON marker_start.id = marker_end.id '
          'LEFT JOIN (SELECT * FROM StringTable) AS strings ON strings._id_ = marker_start.name')
        self.pd_ops = pd.read_sql(
            'SELECT * FROM markers WHERE operator != "Batch- Forward" AND operator != "Batch- Backward"', self.conn)
        self.conn.execute('DROP TABLE markers')
        if self.micro_sec: convert_micro_sec(self.pd_ops)

    def map_kernel_launch_and_execute(self):
        '''
        Correlate kernel launch with execute by correlationId.

        Columns in pd_launch_and_execute:
        - correlationId
        - launch_start
        - launch_end
        - kernel_start
        - kernel_end
        - kernel_name
        - kernel_global_index
        '''
        self.pd_launch_and_execute = pd.read_sql(
            'SELECT launch.correlationId AS correlationId, launch.start AS launch_start, '
            'launch.end AS launch_end, kernel.start AS kernel_start, kernel.end AS kernel_end, strings.value AS kernel_name '
            'FROM (SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE cbid = 211) AS launch '
            'LEFT JOIN (SELECT correlationId, start, end, name AS id FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) AS kernel '
            'ON kernel.correlationId = launch.correlationId '
            'LEFT JOIN (SELECT * FROM StringTable) AS strings ON strings._id_ = kernel.id',
            self.conn)
        if self.micro_sec: convert_micro_sec(self.pd_launch_and_execute)
        self.pd_launch_and_execute['kernel_global_index'] = self.pd_launch_and_execute.index

    def calc_kernel_latency_and_duration(self):
        self.pd_launch_and_execute['kernel_latency'] = \
            self.pd_launch_and_execute['kernel_start'] - self.pd_launch_and_execute['launch_start']
        self.pd_launch_and_execute['kernel_duration'] = \
            self.pd_launch_and_execute['kernel_end'] - self.pd_launch_and_execute['kernel_start']

    def map_operator_and_kernel(self):
        '''
        Classify each CudaLaunchKernel to the operator it belongs to by time range inclusion relation.

        SQL commands and pandas row-wise iteration will be slow here for our purpose.
        '''
        col_launch_and_execute = self.pd_launch_and_execute.columns
        add_cols = col_launch_and_execute.tolist()
        for col in add_cols:
            self.pd_ops[col] = ''
        list_ops = self.pd_ops.to_numpy().tolist()
        col_ops = self.pd_ops.columns
        list_launch_and_execute = self.pd_launch_and_execute.to_numpy().tolist()
        op_start_index = col_ops.get_loc('op_start')
        op_end_index = col_ops.get_loc('op_end')
        launch_start_index = col_launch_and_execute.get_loc('launch_start')
        launch_end_index = col_launch_and_execute.get_loc('launch_end')

        list_final_ops = []
        i = j = 0
        has_entry = False
        while i < len(list_ops) and j < len(list_launch_and_execute):
            op = list_ops[i]
            launch_and_execute = list_launch_and_execute[j]
            op_start, op_end = op[op_start_index], op[op_end_index]
            launch_start, launch_end = \
                launch_and_execute[launch_start_index], launch_and_execute[launch_end_index]
            if launch_end <= op_start:
                j += 1
            elif op_start <= launch_start <= op_end:
                op[-len(add_cols):] = launch_and_execute
                j += 1
                has_entry = True
                list_final_ops.append(op[::])
            else:
                if not has_entry:
                    list_final_ops.append(op[::])
                i += 1
                has_entry = False

        self.pd_kernel_level = pd.DataFrame(list_final_ops, columns=col_ops)

    def set_time_offset_to_zero(self):
        '''
        Set the shown time to be relative to the time_offset.

        After this, the timestamp in each column will
        be consistent with that shown in visual profiler.
        '''
        if self.time_offset == 0: return
        df = self.pd_kernel_level if self.pd_kernel_level is not None else self.pd_launch_and_execute
        for key in df.columns.tolist():
            if 'start' in key or 'end' in key:
                self.pd_kernel_level[key] = self.pd_kernel_level[key].map(
                    lambda x: int(x) - self.time_offset if x != '' else '')

    def reduce_to_op_level(self):
        '''
        columns:
        - step
        - operator
        - num_kernels
        - start
        - overall_duration
        - kernel_duration
        - util_percentile
        '''
        raw_list = self.pd_kernel_level.to_numpy().tolist()
        cols = self.pd_kernel_level.columns
        ks = cols.get_loc('kernel_start')
        ke = cols.get_loc('kernel_end')
        kd = cols.get_loc('kernel_duration')
        op_level_rows = []
        op_level_row = []
        for row in raw_list:
            if not row[kd]: continue
            step, op = row[:2]
            duration = row[kd]
            start = row[ks]
            end = row[ke]
            if not op_level_row:
                op_level_row = [step, op, 1, start, end, duration]
            elif step == op_level_row[0] and op == op_level_row[1]:
                op_level_row[2] += 1        # 2: num_kernels
                op_level_row[4] = end       # 4: end
                op_level_row[5] += duration # 5: duration
            else:
                try:
                    op_level_row[4] -= op_level_row[3]  # 4: end -> overall_duration
                    percentile = op_level_row[5] / op_level_row[4] * 100
                    op_level_rows.append(op_level_row + [percentile])
                    op_level_row = [step, op, 1, start, end, duration]
                except Exception as e:
                    print(op_level_row)
                    raise e
        if op_level_row:
            op_level_row[4] -= op_level_row[3]  # 4: end -> overall_duration
            percentile = op_level_row[5] / op_level_row[4] * 100
            op_level_rows.append(op_level_row + [percentile])
        cols = pd.Index(['step', 'operator', 'num_kernels', 'start', 'kernel_duration', 'overall_duration', 'util_percentile'])
        self.pd_op_level = pd.DataFrame(op_level_rows, columns=cols)

    def process(self):
        self.map_kernel_launch_and_execute()
        self.calc_kernel_latency_and_duration()
        if self.demangle:
            self.demangle_kernel_names()
        if self.with_nvtx:
            self.extract_ort_operators()
            self.tag_steps()
            self.map_operator_and_kernel()

        if self.set_time_offset:
            self.get_time_offset()
            self.set_time_offset_to_zero()

    def export_info_to_csv(self, start=0, end=None, kernel_level=True, op_level=True):
        if end is None: end = max(self.pd_kernel_level['step'].to_numpy().tolist())
        to_export = {}
        if kernel_level: to_export['kernel'] = self.pd_kernel_level
        if op_level: to_export['op'] = self.pd_op_level
        for key, df in to_export.items():
            pd_slice = df[(df.step >= start) & (df.step <= end)]
            save_path = os.path.join(self.path, f'{self.name}_{key}.csv')
            pd_slice.to_csv(save_path, index=False, float_format='%.3f')

    def average_over_steps(self, start=0, end=None, onnx_file=None, export=True):
        '''
        columns:
        - operator
        - num_kernels
        - kernel_duration
        - overall_duration
        - util_percentile
        - (call_stack)
        '''
        steps = self.pd_op_level['step'].to_numpy().tolist()
        min_step = max(min(steps), start)
        max_step = max(steps)
        if end is not None: max_step = min(max_step, end)
        self.pd_op_avg = self.pd_op_level[self.pd_op_level.step == min_step].reset_index(drop=True)

        for i in range(min_step + 1, max_step + 1):
            tmp_df = self.pd_op_level[self.pd_op_level.step == i].reset_index(drop=True)
            self.pd_op_avg = self.pd_op_avg + tmp_df
        self.pd_op_avg['operator'] = tmp_df['operator']
        for key in ('num_kernels', 'kernel_duration', 'overall_duration', 'util_percentile'):
            self.pd_op_avg[key] = self.pd_op_avg[key].apply(lambda x: x / (max_step - min_step + 1))
        self.pd_op_avg = self.pd_op_avg.drop(['step', 'start'], axis=1)

        if onnx_file is not None:
            import onnx
            model = onnx.load(onnx_file)
            map_op_stack = {node.name: node.doc_string for node in model.graph.node}
            try:
                self.pd_op_avg['call_stack'] = self.pd_op_avg['operator'].map(
                    lambda x: map_op_stack.get(x.split('(')[-1][:-1], '')
                )
            except Exception as e:
                print(self.pd_op_avg['operator'])
                raise e
        if export:
            save_path = os.path.join(self.path, f'{self.name}_avg.csv')
            self.pd_op_avg.to_csv(save_path, index=False, float_format='%.3f')



class NsysParser(NvvpParser):
    def __init__(self, nvvp_file, with_nvtx=True):
        super().__init__(nvvp_file, with_nvtx, False, False, False)
        self.process()

    def extract_ort_operators(self):
        '''
        Extract ORT operator marker information.

        Columns in operators (pd_ops):
        - operator ('Batch- Forward' and 'Batch- Backward' excluded)
        - op_start
        - op_end
        '''
        self.pd_ops = pd.read_sql(
            'SELECT text AS operator, start AS op_start, end AS op_end FROM NVTX_EVENTS '
            'WHERE text != "Batch- Forward" AND text != "Batch- Backward"',
            self.conn
        )
        if self.micro_sec: convert_micro_sec(self.pd_ops)

    def map_kernel_launch_and_execute(self):
        '''
        Correlate kernel launch with execute by correlationId.

        Columns in pd_launch_and_execute:
        - correlationId
        - launch_start
        - launch_end
        - kernel_start
        - kernel_end
        - kernel_name
        - kernel_global_index
        '''
        self.pd_launch_and_execute = pd.read_sql(
            'SELECT launch.correlationId AS correlationId, launch.start AS launch_start, '
            'launch.end AS launch_end, kernel.start AS kernel_start, kernel.end AS kernel_end, strings.value AS kernel_name '
            'FROM (SELECT correlationId, start, end, demangledName AS id FROM CUPTI_ACTIVITY_KIND_KERNEL) AS kernel '
            'LEFT JOIN (SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME) AS launch '
            'ON kernel.correlationId = launch.correlationId '
            'LEFT JOIN (SELECT * FROM StringIds) AS strings ON strings.id = kernel.id',
            self.conn)
        if self.micro_sec: convert_micro_sec(self.pd_launch_and_execute)
        self.pd_launch_and_execute['kernel_global_index'] = self.pd_launch_and_execute.index


def demangle(raw_name):
    import cxxfilt
    try:
        return cxxfilt.demangle(raw_name)
    except:
        # print(f'{raw_name} cannot be demangled.')
        return raw_name

def convert_micro_sec(df):
    for key in df.columns.tolist():
        if key.endswith('start') or key.endswith('end'):
            df[key] = df[key].map(lambda x: x / 1000 if x != '' else '')

def all_kernels_aligned(parser_nvtx, parser_nonvtx):
    kernel_list_1 = parser_nvtx.pd_launch_and_execute['kernel_name'].to_numpy().tolist()
    kernel_list_2 = parser_nonvtx.pd_launch_and_execute['kernel_name'].to_numpy().tolist()
    return len(kernel_list_1) == len(kernel_list_2) and all(
        [a == b for a, b in zip(kernel_list_1, kernel_list_2)])

def map_op_with_kernel_launch(parser_nvtx, parser_nonvtx):
    if parser_nonvtx.pd_kernel_level is not None: return
    op_kernel_list = parser_nvtx.pd_kernel_level[['step', 'operator', 'kernel_global_index']].to_numpy().tolist()
    op_kernel_map = {indx: [step, op] for step, op, indx in op_kernel_list if indx != ''}
    raw_kernel_list = parser_nonvtx.pd_launch_and_execute.to_numpy().tolist()
    index_col = parser_nonvtx.pd_launch_and_execute.columns.get_loc('kernel_global_index')
    new_indices = pd.Index(['step', 'operator'])
    cols = new_indices.append(parser_nonvtx.pd_launch_and_execute.columns)

    total_info = []
    for row in raw_kernel_list:
        indx = row[index_col]
        if indx in op_kernel_map:
            total_info.append(op_kernel_map[indx] + row)
    parser_nonvtx.pd_kernel_level = pd.DataFrame(total_info, columns=cols)

def main(file_nvtx, file_nonvtx, nsys=False, onnx_file=None):
    parser = NsysParser if nsys else NvvpParser
    parser_nvtx = parser(file_nvtx)
    parser_nonvtx = parser(file_nonvtx, False)

    assert all_kernels_aligned(parser_nvtx, parser_nonvtx)
    map_op_with_kernel_launch(parser_nvtx, parser_nonvtx)

    parser_nonvtx.reduce_to_op_level()
    parser_nonvtx.average_over_steps(onnx_file=onnx_file, export=True)
    parser_nonvtx.export_info_to_csv()


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file1', type=str, required=True, help='profiling file (*.nvvp or *.sqlite) with ORT NVTX enabled')
    arg_parser.add_argument('--file2', type=str, required=True, help='profiling file (*.nvvp or *.sqlite) with ORT NVTX disabled')
    arg_parser.add_argument('--nsys', action='store_true', default=False, help='nsys (file *.sqlite) or nvprof (file *.nvvp)')
    arg_parser.add_argument('--onnx', type=str, default=None, help='onnx file of torch exported model, with call stack info')

    args = arg_parser.parse_args()

    main(args.file1, args.file2, args.nsys, args.onnx)
