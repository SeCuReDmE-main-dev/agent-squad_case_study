export class Triplet {
  readonly T: number;
  readonly I: number;
  readonly F: number;

  constructor(T: number, I: number, F: number) {
    this.T = Triplet.validateScore(T, "T");
    this.I = Triplet.validateScore(I, "I");
    this.F = Triplet.validateScore(F, "F");
  }

  private static validateScore(value: number, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(`${name} must be a finite number`);
    }

    if (value < 0 || value > 1) {
      throw new RangeError(`${name} must be between 0 and 1`);
    }

    return value;
  }
}
