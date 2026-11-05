"""Package the course's MiniProject<leader roll>/Report and Code structure."""
import argparse
from pathlib import Path
import re
import zipfile

from scripts.package_heatline import write_code


def package_submission(roll_number, report_dir, output):
    if not re.fullmatch(r'[0-9]+', roll_number):
        raise ValueError('Team leader roll number must contain digits only')
    reports = sorted(p for p in Path(report_dir).iterdir() if p.suffix.lower() in {'.pdf', '.docx', '.pptx'})
    if not any(p.suffix.lower() == '.pptx' for p in reports) or not any(p.suffix.lower() in {'.pdf', '.docx'} for p in reports):
        raise ValueError('Report directory must contain a report and PowerPoint presentation')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'MiniProject{roll_number}'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        write_code(archive, prefix=f'{prefix}/Code')
        for report in reports:
            archive.write(report, f'{prefix}/Report/{report.name}')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Submission archive integrity check failed')
    print(f'{output}: {output.stat().st_size:,} bytes')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roll', required=True)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    package_submission(args.roll, args.report_dir, args.output)
